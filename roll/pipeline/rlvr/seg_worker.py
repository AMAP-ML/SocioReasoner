import torch
import re
import ast
import json
from functools import partial
from sam2.build_sam import build_sam2
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.configs.worker_config import WorkerConfig
from roll.utils.context_managers import state_offload_manger
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.offload_states import OffloadStateType
from roll.models.model_providers import sam2_seg_model_provider
        
def parse_points_from_content(content):
    """
    从内容字符串中提取一个三层嵌套的2D点列表。

    Args:
        content (str): 包含点的字符串，嵌入在 <answer> 标签内。
                       示例: "一些文本 <answer>[[[10,20],[30,40]],[[50,60]]]</answer> 更多文本"

    Returns:
        list: 解析后的列表，格式为 [[[x,y],[x,y]], ...]。
              如果找不到模式或解析失败，则返回空列表。
    """
    # 正则表达式部分无需修改，它能捕获标签内的任何内容
    answer_pattern = r"<answer>(.*?)</answer>"
    
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:
        points_text = answer_match.group(1)
        try:
            # ast.literal_eval 也无需修改，它可以安全地解析多层嵌套的列表
            parsed_data = ast.literal_eval(points_text)
            
            # #############################################################
            # 核心修改：更新验证逻辑以匹配三层嵌套结构
            # #############################################################
            if not isinstance(parsed_data, list):
                # print("顶层结构不是列表")
                return []

            # 使用 all() 和生成器表达式来验证三层结构
            # 1. 检查第一层 (parsed_data) 的所有元素 (group) 是不是列表
            # 2. 检查第二层 (group) 的所有元素 (point) 是不是列表
            # 3. 检查第三层 (point) 是否恰好包含2个数字坐标 (coord)
            is_valid_structure = all(
                isinstance(group, list) and all(
                    isinstance(point, list) and len(point) == 2 and all(
                        isinstance(coord, (int, float)) for coord in point
                    )
                    for point in group
                )
                for group in parsed_data
            )
            
            if not is_valid_structure:
                # print(f"错误: 解析出的数据不是预期的三层嵌套结构。数据: {parsed_data}")
                return []
            
            return parsed_data
            # #############################################################
            # 修改结束
            # #############################################################

        except (ValueError, SyntaxError) as e:
            # print(f"错误: 无法将 '{points_text}' 解析为Python字面量: {e}")
            return []
        except Exception as e:
            # print(f"解析过程中发生未知错误: {e}")
            return []
    else:
        # print("在内容中未找到 <answer> 标签。")
        return []

def parse_points_from_content_v2(content: str) -> list:
    """
    从包含 <answer> 标签的字符串中，提取、解析JSON格式的“坐标点字典”列表，
    并将其转换为三层嵌套的2D点列表。

    Args:
        content (str): 包含JSON数据的字符串，该数据嵌入在 <answer> 标签内。
                       示例: "一些文本 <answer>[{"p1":[1,2], "p2":[3,4]}, {"p3":[5,6]}]</answer> 更多文本"

    Returns:
        list: 解析并转换成功后的列表，格式为 [[[x,y],[x,y]], ...]。
              如果找不到 <answer> 标签，或解析失败，或数据结构不符合要求，则返回空列表。
              示例输出: [[[1,2],[3,4]], [[5,6]]]
    """
    # 步骤 1: 使用正则表达式查找 <answer> 标签内的内容
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)

    if answer_match:
        # 提取标签中的JSON字符串，并去除可能存在的前后空白
        points_text = answer_match.group(1).strip()
        
        try:
            # 步骤 2: 使用 json.loads 解析提取出的字符串
            parsed_data = json.loads(points_text)

            # 步骤 3: 验证解析出的数据结构是否符合预期
            
            # 验证 1: 顶层结构必须是一个列表
            if not isinstance(parsed_data, list):
                return []

            # 验证 2: 列表中的每个元素都必须是包含有效坐标点的字典
            is_valid_structure = all(
                isinstance(obj, dict) and obj and all(
                    isinstance(point, list) and len(point) == 2 and all(
                        isinstance(coord, (int, float)) for coord in point
                    )
                    for point in obj.values()
                )
                for obj in parsed_data
            )
            
            if not is_valid_structure:
                return []
            
            # 核心修改：将 [{"key": [x,y]}, ...] 格式转换为 [[[x,y], ...], ...]
            # 遍历解析后的列表，对每个字典，提取其所有的值（即坐标点）
            # 并将这些坐标点收集到一个新的列表中。
            return [list(obj.values()) for obj in parsed_data]

        except json.JSONDecodeError:
            # 如果 points_text 不是有效的JSON，则解析失败
            return []
        except Exception:
            # 捕获其他任何在验证过程中可能发生的意外错误
            return []
    else:
        # 如果在 content 中未找到 <answer> 标签
        return []

def parse_visual_prompt_from_json_s1(content: str) -> List[Dict[str, Any]]:
    """
    从LLM生成的JSON字符串输出中解析出视觉提示信息。

    Args:
        content (str): 一个包含物体提示信息的JSON格式字符串。
                       例如: '[{"bbox_2d": [10,100,200,210], "point_2d": [[70,180,0],[20,200,1]]}, ...]'

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个物体的提示，
                              包含 'box', 'points', 'labels' 键，以适配后续处理。
    """
    parsed_objects = []
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)

    if answer_match:
        # 提取标签中的JSON字符串，并去除可能存在的前后空白
        points_text = answer_match.group(1).strip()
        try:
            # 使用 json.loads 将JSON字符串安全地转换为Python对象（列表）
            data = json.loads(points_text)
            
            # 确保解析出的数据是列表类型
            if not isinstance(data, list):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []

            # 遍历列表中的每个对象
            for obj in data:
                try:
                    # 确保对象是字典并包含必要的键
                    if not isinstance(obj, dict):
                        print(f"Warning: Skipped malformed object in JSON. Object: {obj}")
                        continue

                    box = obj.get("bbox_2d", [])

                    # 确保所有解析出的数据格式正确
                    if isinstance(box, list) and len(box) == 4:
                        parsed_objects.append({
                            "box": box,
                        })
                    else:
                        print(f"Warning: Skipped object with incorrect internal data types. Object: {obj}")
                except Exception as e:
                    # 如果解析过程中发生任何异常，则跳过该对象
                    print(f"Error parsing object: {e}. Object: {obj}")

        except json.JSONDecodeError as e:
            # 如果字符串不是有效的JSON，则会抛出异常
            print(f"Error parsing JSON string: {e}. Content: {content}")
        
    return parsed_objects

def parse_visual_prompt_from_json_s2(content: str) -> List[Dict[str, Any]]:
    """
    从LLM生成的JSON字符串输出中解析出视觉提示信息。

    Args:
        content (str): 一个包含物体提示信息的JSON格式字符串。
                       例如: '[{"bbox_2d": [10,100,200,210], "point_2d": [[70,180,0],[20,200,1]]}, ...]'

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个物体的提示，
                              包含 'box', 'points', 'labels' 键，以适配后续处理。
    """
    parsed_objects = []
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:
        # 提取标签中的JSON字符串，并去除可能存在的前后空白
        points_text = answer_match.group(1).strip()
        try:
            # 使用 json.loads 将JSON字符串安全地转换为Python对象（列表）
            data = json.loads(points_text)
            
            # 确保解析出的数据是列表类型
            if not isinstance(data, list):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []

            # 遍历列表中的每个对象
            for obj in data:
                try:
                    # 确保对象是字典并包含必要的键
                    if not isinstance(obj, dict):
                        print(f"Warning: Skipped malformed object in JSON. Object: {obj}")
                        continue

                    box = obj.get("bbox_2d", [])
                    point_data = obj.get("points", [])
                    
                    points = [[p[0], p[1]] for p in point_data]  # 提取 [x, y]
                    labels = np.ones(len(points), dtype=int)
                    labels = labels.tolist()
                    # label to list

                    # 确保所有解析出的数据格式正确
                    if isinstance(box, list) and isinstance(points, list) and isinstance(labels, list) and len(box) == 4:
                        parsed_objects.append({
                            "box": box,
                            "points": points,
                            "labels": labels
                        })
                    else:
                        print(f"Warning: Skipped object with incorrect internal data types. Object: {obj}")
                except Exception as e:
                    # 如果解析过程中发生任何异常，则跳过该对象
                    print(f"Error parsing object: {e}. Object: {obj}")

        except json.JSONDecodeError as e:
            # 如果字符串不是有效的JSON，则会抛出异常
            print(f"Error parsing JSON string: {e}. Content: {content}")
    
    return parsed_objects

def parse_visual_prompt_from_json_s2_old(content: str) -> List[Dict[str, Any]]:
    """
    从LLM生成的JSON字符串输出中解析出视觉提示信息。

    Args:
        content (str): 一个包含物体提示信息的JSON格式字符串。
                       例如: '[{"bbox_2d": [10,100,200,210], "point_2d": [[70,180,0],[20,200,1]]}, ...]'

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个物体的提示，
                              包含 'box', 'points', 'labels' 键，以适配后续处理。
    """
    parsed_objects = []
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:
        # 提取标签中的JSON字符串，并去除可能存在的前后空白
        points_text = answer_match.group(1).strip()
        try:
            # 使用 json.loads 将JSON字符串安全地转换为Python对象（列表）
            data = json.loads(points_text)
            
            # 确保解析出的数据是列表类型
            if not isinstance(data, list):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []

            # 遍历列表中的每个对象
            for obj in data:
                try:
                    # 确保对象是字典并包含必要的键
                    if not isinstance(obj, dict):
                        print(f"Warning: Skipped malformed object in JSON. Object: {obj}")
                        continue

                    box = obj.get("bbox_2d", [])
                    point_data = obj.get("point_2d", [])
                    
                    points = [[p[0], p[1]] for p in point_data]  # 提取 [x, y]
                    labels = [p[2] for p in point_data]        # 提取 label

                    # 确保所有解析出的数据格式正确
                    if isinstance(box, list) and isinstance(points, list) and isinstance(labels, list) and len(box) == 4:
                        parsed_objects.append({
                            "box": box,
                            "points": points,
                            "labels": labels
                        })
                    else:
                        print(f"Warning: Skipped object with incorrect internal data types. Object: {obj}")
                except Exception as e:
                    # 如果解析过程中发生任何异常，则跳过该对象
                    print(f"Error parsing object: {e}. Object: {obj}")

        except json.JSONDecodeError as e:
            # 如果字符串不是有效的JSON，则会抛出异常
            print(f"Error parsing JSON string: {e}. Content: {content}")
    
    return parsed_objects

def parse_visual_prompt_from_json_s2_sat(content: str, bbox_text: str) -> List[Dict[str, Any]]:
    """
    从LLM生成的JSON字符串输出中解析出视觉提示信息。

    Args:
        content (str): 一个包含物体提示信息的JSON格式字符串。
                       例如: '[{"bbox_2d": [10,100,200,210], "point_2d": [[70,180,0],[20,200,1]]}, ...]'

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个物体的提示，
                              包含 'box', 'points', 'labels' 键，以适配后续处理。
    """
    parsed_objects = []
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:
        # 提取标签中的JSON字符串，并去除可能存在的前后空白
        points_text = answer_match.group(1).strip()
        try:
            # 使用 json.loads 将JSON字符串安全地转换为Python对象（列表）
            data = json.loads(points_text)
            bbox_data = json.loads(bbox_text)
            
            # 确保解析出的数据是列表类型
            if not isinstance(data, list):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []
            if not isinstance(bbox_data, list):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []
            if len(data) != len(bbox_data):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []
            # 遍历列表中的每个对象
            for obj, bbox in zip(data, bbox_data):
                try:
                    # 确保对象是字典并包含必要的键
                    if not isinstance(obj, dict):
                        print(f"Warning: Skipped malformed object in JSON. Object: {obj}")
                        continue

                    box = bbox.get("bbox_2d", [])
                    point_data = obj.get("point_2d", [])
                    
                    points = [[p[0], p[1]] for p in point_data]  # 提取 [x, y]
                    labels = [p[2] for p in point_data]        # 提取 label

                    # 确保所有解析出的数据格式正确
                    if isinstance(box, list) and isinstance(points, list) and isinstance(labels, list) and len(box) == 4:
                        parsed_objects.append({
                            "box": box,
                            "points": points,
                            "labels": labels
                        })
                    else:
                        print(f"Warning: Skipped object with incorrect internal data types. Object: {obj}")
                except Exception as e:
                    # 如果解析过程中发生任何异常，则跳过该对象
                    print(f"Error parsing object: {e}. Object: {obj}")

        except json.JSONDecodeError as e:
            # 如果字符串不是有效的JSON，则会抛出异常
            print(f"Error parsing JSON string: {e}. Content: {content}")
    
    return parsed_objects

def parse_visual_prompt_from_content_segr1(content: str) -> List[Dict[str, Any]]:
    """
    从LLM的文本输出中解析出SAM的提示信息（边界框、点、标签）。

    Args:
        content (str): LLM生成的包含视觉提示的字符串。
                       例如："<bbox>[...],<points>[[...]],<labels>[...]</bbox>, ..."

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个物体的提示，
                              包含 'box', 'points', 'labels' 键。
    """
    # 正则表达式，用于匹配每个物体的 bbox, points, 和 labels
    # 使用非贪婪匹配 (.*?) 来确保正确处理多个物体的情况
    pattern = re.compile(r"<bbox>(.*?)</bbox>, <points>(.*?)</points>, <labels>(.*?)</labels>")
    
    matches = pattern.findall(content)
    
    parsed_objects = []
    for box_str, points_str, labels_str in matches:
        try:
            # 使用 ast.literal_eval 安全地将字符串转换为Python对象（列表）
            box = ast.literal_eval(box_str)
            points = ast.literal_eval(points_str)
            labels = ast.literal_eval(labels_str)
            
            # 确保解析出的数据是列表类型
            if isinstance(box, list) and isinstance(points, list) and isinstance(labels, list):
                parsed_objects.append({
                    "box": box,
                    "points": points,
                    "labels": labels
                })
            else:
                # 如果数据格式不正确，可以选择跳过或记录日志
                print(f"Warning: Skipped malformed prompt data. Box: {box_str}, Points: {points_str}")

        except (ValueError, SyntaxError) as e:
            # 如果字符串不是有效的Python字面量，则会抛出异常
            print(f"Error parsing prompt string: {e}. Content part: {box_str}, {points_str}, {labels_str}")
            continue
            
    return parsed_objects


def parse_visual_prompt_from_content_samr1(content: str) -> List[Dict[str, Any]]:
    """
    从LLM的文本输出中解析出SAM的提示信息。
    此函数会先从文本中提取被 <answer>...</answer> 标签包裹的JSON字符串，然后再进行解析。

    此版本处理的输入是一个包含JSON内容的文本块。
    "points"列表中的每个元素格式为 [x, y, label]。
    函数会将其拆分为独立的 "points" ([x, y]) 和 "labels" 列表以保持输出格式的统一。

    Args:
        content (str): LLM生成的包含视觉提示的字符串，其中JSON部分被<answer>标签包裹。
                       例如：'<think>...</think><answer>{"bbox": [248,218,395,300], "points": [[360,237,1]]}</answer>'

    Returns:
        List[Dict[str, Any]]: 一个字典列表。如果解析成功，列表中将包含一个字典，
                               该字典代表一个物体的提示，包含 'box', 'points', 'labels' 键。
                               如果解析失败，则返回空列表。
    """
    parsed_objects = []
    try:
        # 1. 使用正则表达式从content中提取<answer>标签内的内容
        # re.DOTALL 使得 '.' 可以匹配包括换行符在内的任意字符
        match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if not match:
            # 如果没有找到<answer>标签，则打印警告并返回空列表
            print(f"Warning: Could not find <answer> tag in the content. Content: {content}")
            return []
        
        # 获取第一个捕获组的内容，并去除首尾的空白字符
        json_content = match.group(1).strip()

        # 2. 将提取出的JSON字符串解析为Python字典
        data = json.loads(json_content)

        # 从字典中获取bbox
        box = data.get("bbox")
        # 从字典中获取带有标签的点
        points_with_labels = data.get("points")

        # 检查关键数据是否存在且格式基本正确
        # 注意：这里允许 box 或 points_with_labels 为 None 或空列表
        if box is not None and not isinstance(box, list):
             print(f"Warning: Skipped malformed JSON data. 'bbox' is not a list. Content: {content}")
             return []
        if points_with_labels is not None and not isinstance(points_with_labels, list):
            print(f"Warning: Skipped malformed JSON data. 'points' is not a list. Content: {content}")
            return []

        # 初始化用于存储最终结果的列表
        points_xy = []
        labels = []

        # 遍历带有标签的点，将其拆分为坐标和标签
        if points_with_labels: # 仅当列表不为None且不为空时处理
            for point_data in points_with_labels:
                if isinstance(point_data, list) and len(point_data) == 3:
                    # 前两个元素是坐标
                    points_xy.append([point_data[0], point_data[1]])
                    # 第三个元素是标签
                    labels.append(point_data[2])
                else:
                    print(f"Warning: Skipped malformed point data. Expected [x, y, label]. Got: {point_data}")
        
        # 将解析出的数据构造成最终的字典格式并添加到结果列表中
        # 即使box或points为空，也创建一个有效的对象结构
        parsed_objects.append({
            "box": box if box is not None else [],
            "points": points_xy,
            "labels": labels
        })

    except json.JSONDecodeError as e:
        # 如果字符串不是有效的JSON，则会抛出异常
        print(f"Error parsing JSON string from <answer> tag: {e}. Content: {content}")
    except (TypeError, AttributeError) as e:
        # 捕获其他可能的错误，例如 .get() 之后的对象不是预期的类型
        print(f"Error processing parsed JSON data: {e}. Content: {content}")
        
    return parsed_objects

class SegWorker(Worker):
    """
    一个专用于SAM（Segment Anything Model）模型推理的Worker。
    它只包含模型加载和推理的逻辑，没有训练过程。
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        # 策略对象，将封装SAM模型和推理逻辑
        self.strategy: Optional[InferenceStrategy] = None
        self.tokenizer = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config, tokenizer=None):
        """
        初始化Worker，加载SAM模型。
        此方法会被Cluster广播到所有SAMWorker实例上。
        """
        super().initialize(pipeline_config)
        
        self.tokenizer = tokenizer
        # breakpoint()

        # 使用工厂函数创建专门用于SAM的策略
        self.strategy = create_strategy(worker=self)

        # 初始化策略，这里需要一个SAM模型的提供者(provider)
        # 您需要根据项目实现一个 sam_model_provider
        self.strategy.initialize(model_provider=sam2_seg_model_provider)
        

        # 初始化后立即卸载模型权重，释放GPU显存
        self.strategy.offload_states()
        self.logger.info(f"{self.worker_name} (SAMWorker) initialized")

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment(self, data: DataProto) -> DataProto:
        """
        执行SAM模型的分割推理任务。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'prompts' 等
                              SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}
        # breakpoint()
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)

        # print("!!!!!!!!!!!!!!!!!!!worker input data!!!!!!!!!!!!!!!!!!")
        # print(data)
        
        points_list = []
        for response in response_text_list:            
            # coords_list 的例子: [[10, 20], [30, 40]] 或 []
            # 检查response类型
            
            coords_list = parse_points_from_content(response)
            
            sam_prompt_list = []
            for coords in coords_list:
                point_coords = np.array(coords)
                
                point_labels = np.ones(point_coords.shape[0], dtype=int)

                sam_prompt_dict = {
                    'point_coords': point_coords,
                    'point_labels': point_labels
                }
                sam_prompt_list.append(sam_prompt_dict)

            points_list.append(sam_prompt_list)
            
        # print(points_list)
        data.non_tensor_batch["visual_prompt"] = np.array(points_list, dtype=object) 
        # 使用状态管理器，在计算时自动将模型加载到GPU，计算后卸载
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",
            # 推理只需要加载模型参数
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            # 1. 将输入数据移动到CUDA设备
            data = data.to("cuda")

            # 2. 调用策略对象执行核心推理逻辑
            #    假设您的SAMStrategy中有一个名为 'segment' 的方法
            output_batch = self.strategy.segment(batch=data)
            # breakpoint()
            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list

            # 4. 将输入和输出数据移回CPU，释放GPU显存
            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data

        
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_v2(self, data: DataProto) -> DataProto:
        """
        执行SAM模型的分割推理任务。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'prompts' 等
                              SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}
        # breakpoint()
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)

        # print("!!!!!!!!!!!!!!!!!!!worker input data!!!!!!!!!!!!!!!!!!")
        # print(data)
        
        points_list = []
        for response in response_text_list:            
            # coords_list 的例子: [[10, 20], [30, 40]] 或 []
            # 检查response类型
            
            coords_list = parse_points_from_content_v2(response)
            
            sam_prompt_list = []
            for coords in coords_list:
                point_coords = np.array(coords)
                
                point_labels = np.ones(point_coords.shape[0], dtype=int)

                sam_prompt_dict = {
                    'point_coords': point_coords,
                    'point_labels': point_labels
                }
                sam_prompt_list.append(sam_prompt_dict)

            points_list.append(sam_prompt_list)
            
        # print(points_list)
        data.non_tensor_batch["visual_prompt"] = np.array(points_list, dtype=object) 
        # 使用状态管理器，在计算时自动将模型加载到GPU，计算后卸载
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",
            # 推理只需要加载模型参数
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            # 1. 将输入数据移动到CUDA设备
            data = data.to("cuda")

            # 2. 调用策略对象执行核心推理逻辑
            #    假设您的SAMStrategy中有一个名为 'segment' 的方法
            output_batch = self.strategy.segment(batch=data)
            # breakpoint()
            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list

            # 4. 将输入和输出数据移回CPU，释放GPU显存
            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data
    
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_v3(self, data: "DataProto") -> "DataProto":
        """
        执行SAM模型的分割推理任务。
        此函数已更新，可以正确解析LLM输出并构建包含多种提示的SAM输入。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'responses' 等
                            SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}
        # 从data.batch中解码LLM的文本响应
        # breakpoint()
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        stage_flag_list = data.non_tensor_batch["stage_flag"]
        prompts_for_batch = []
        for response, stage_flag in zip(response_text_list, stage_flag_list):
            # 1. 解析单个响应字符串，获取所有物体的提示信息
            # parsed_objects 的例子: [{'box': [x,y,x,y], 'points': [[x,y]], 'labels': [1]}, ...]
            if stage_flag == 1:
                parsed_objects = parse_visual_prompt_from_json_s1(response)
            else:
                parsed_objects = parse_visual_prompt_from_json_s2(response)
            
            sam_prompt_list_for_image = []
            # 2. 遍历这张图片中所有被识别出的物体
            for obj_prompts in parsed_objects:

                sam_prompt_dict = {}
                try:
                    # 检查并添加边界框提示
                    if 'box' in obj_prompts and obj_prompts['box']:
                        # SAM 通常期望 box 是 (1, 4) 的 numpy 数组
                        if len(obj_prompts['box']) == 4:
                            sam_prompt_dict['box'] = np.array(obj_prompts['box'])

                    # 检查并添加点和标签提示
                    if 'points' in obj_prompts and obj_prompts['points']:
                        point_coords = np.array(obj_prompts['points'])
                        point_labels = np.array(obj_prompts['labels'])
                        
                        # 确保点和标签数量匹配, 确保每个point的长度为2
                        if point_coords.shape[0] == point_labels.shape[0] and point_coords.shape[1] == 2 and len(point_labels.shape) == 1:
                            sam_prompt_dict['point_coords'] = point_coords
                            sam_prompt_dict['point_labels'] = point_labels
                        else:
                            print(f"Warning: Mismatch between points ({point_coords.shape[0]}) and labels ({point_labels.shape[0]}). Skipping points for this object.")
                except Exception as e:
                    pass
                
                if sam_prompt_dict:
                    sam_prompt_list_for_image.append(sam_prompt_dict)

            prompts_for_batch.append(sam_prompt_list_for_image)
            
            
        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)
        
        # --- 后续的推理流程保持不变 ---

        # 使用状态管理器，在计算时自动将模型加载到GPU，计算后卸载
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",
            # 推理只需要加载模型参数
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            # 1. 将输入数据移动到CUDA设备
            data = data.to("cuda")

            # 2. 调用策略对象执行核心推理逻辑
            #    您的SAMStrategy中的 'segment' 方法现在会收到我们精心构建的 prompts_for_batch
            output_batch = self.strategy.segment(batch=data)
            
            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list

            # 4. 将输入和输出数据移回CPU，释放GPU显存
            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data
    
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_v4_map(self, data: "DataProto") -> "DataProto":
        """
        执行SAM模型的分割推理任务。
        此函数已更新，可以正确解析LLM输出并构建包含多种提示的SAM输入。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'responses' 等
                            SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}
        # 从data.batch中解码LLM的文本响应
        # breakpoint()
        response_text_list = self.tokenizer.batch_decode(data.batch["map_responses"], skip_special_tokens=True)
        prompts_for_batch = []
        for response in response_text_list:
            # 1. 解析单个响应字符串，获取所有物体的提示信息
            # parsed_objects 的例子: [{'box': [x,y,x,y], 'points': [[x,y]], 'labels': [1]}, ...]
            parsed_objects = parse_visual_prompt_from_json_s2(response)
            
            sam_prompt_list_for_image = []
            # 2. 遍历这张图片中所有被识别出的物体
            for obj_prompts in parsed_objects:

                sam_prompt_dict = {}
                try:
                    # 检查并添加边界框提示
                    if 'box' in obj_prompts and obj_prompts['box']:
                        # SAM 通常期望 box 是 (1, 4) 的 numpy 数组
                        if len(obj_prompts['box']) == 4:
                            sam_prompt_dict['box'] = np.array(obj_prompts['box'])

                    # 检查并添加点和标签提示
                    if 'points' in obj_prompts and obj_prompts['points']:
                        point_coords = np.array(obj_prompts['points'])
                        point_labels = np.array(obj_prompts['labels'])
                        
                        # 确保点和标签数量匹配, 确保每个point的长度为2
                        if point_coords.shape[0] == point_labels.shape[0] and point_coords.shape[1] == 2 and len(point_labels.shape) == 1:
                            sam_prompt_dict['point_coords'] = point_coords
                            sam_prompt_dict['point_labels'] = point_labels
                        else:
                            print(f"Warning: Mismatch between points ({point_coords.shape[0]}) and labels ({point_labels.shape[0]}). Skipping points for this object.")
                except Exception as e:
                    pass
                
                if sam_prompt_dict:
                    sam_prompt_list_for_image.append(sam_prompt_dict)

            prompts_for_batch.append(sam_prompt_list_for_image)
            
            
        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)
        
        # --- 后续的推理流程保持不变 ---

        # 使用状态管理器，在计算时自动将模型加载到GPU，计算后卸载
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",
            # 推理只需要加载模型参数
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            # 1. 将输入数据移动到CUDA设备
            data = data.to("cuda")

            # 2. 调用策略对象执行核心推理逻辑
            #    您的SAMStrategy中的 'segment' 方法现在会收到我们精心构建的 prompts_for_batch
            output_batch = self.strategy.segment(batch=data)
            
            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list

            # 4. 将输入和输出数据移回CPU，释放GPU显存
            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data
    
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_v4_sat(self, data: "DataProto") -> "DataProto":
        """
        执行SAM模型的分割推理任务。
        此函数已更新，可以正确解析LLM输出并构建包含多种提示的SAM输入。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'responses' 等
                            SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}
        # 从data.batch中解码LLM的文本响应
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        prompts_for_batch = []
        for response in response_text_list:
            # 1. 解析单个响应字符串，获取所有物体的提示信息
            # parsed_objects 的例子: [{'box': [x,y,x,y], 'points': [[x,y]], 'labels': [1]}, ...]
            parsed_objects = parse_visual_prompt_from_json_s2(response)
            
            sam_prompt_list_for_image = []
            # 2. 遍历这张图片中所有被识别出的物体
            for obj_prompts in parsed_objects:

                sam_prompt_dict = {}
                try:
                    # 检查并添加边界框提示
                    if 'box' in obj_prompts and obj_prompts['box']:
                        # SAM 通常期望 box 是 (1, 4) 的 numpy 数组
                        if len(obj_prompts['box']) == 4:
                            sam_prompt_dict['box'] = np.array(obj_prompts['box'])

                    # 检查并添加点和标签提示
                    if 'points' in obj_prompts and obj_prompts['points']:
                        point_coords = np.array(obj_prompts['points'])
                        point_labels = np.array(obj_prompts['labels'])
                        
                        # 确保点和标签数量匹配, 确保每个point的长度为2
                        if point_coords.shape[0] == point_labels.shape[0] and point_coords.shape[1] == 2 and len(point_labels.shape) == 1:
                            sam_prompt_dict['point_coords'] = point_coords
                            sam_prompt_dict['point_labels'] = point_labels
                        else:
                            print(f"Warning: Mismatch between points ({point_coords.shape[0]}) and labels ({point_labels.shape[0]}). Skipping points for this object.")
                except Exception as e:
                    pass
                
                if sam_prompt_dict:
                    sam_prompt_list_for_image.append(sam_prompt_dict)

            prompts_for_batch.append(sam_prompt_list_for_image)
            
            
        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)
        
        # --- 后续的推理流程保持不变 ---

        # 使用状态管理器，在计算时自动将模型加载到GPU，计算后卸载
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",
            # 推理只需要加载模型参数
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            # 1. 将输入数据移动到CUDA设备
            data = data.to("cuda")

            # 2. 调用策略对象执行核心推理逻辑
            #    您的SAMStrategy中的 'segment' 方法现在会收到我们精心构建的 prompts_for_batch
            output_batch = self.strategy.segment(batch=data)
            
            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list

            # 4. 将输入和输出数据移回CPU，释放GPU显存
            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_segr1(self, data: "DataProto") -> "DataProto":
        """
        执行SAM模型的分割推理任务。
        此函数已更新，可以正确解析LLM输出并构建包含多种提示的SAM输入。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'responses' 等
                            SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}
        # 从data.batch中解码LLM的文本响应
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        prompts_for_batch = []
        for response in response_text_list:
            # 1. 解析单个响应字符串，获取所有物体的提示信息
            # parsed_objects 的例子: [{'box': [x,y,x,y], 'points': [[x,y]], 'labels': [1]}, ...]
            parsed_objects = parse_visual_prompt_from_content_segr1(response)
            
            sam_prompt_list_for_image = []
            # 2. 遍历这张图片中所有被识别出的物体
            for obj_prompts in parsed_objects:

                sam_prompt_dict = {}
                try:
                    # 检查并添加边界框提示
                    if 'box' in obj_prompts and obj_prompts['box']:
                        # SAM 通常期望 box 是 (1, 4) 的 numpy 数组
                        if len(obj_prompts['box']) == 4:
                            sam_prompt_dict['box'] = np.array(obj_prompts['box'])

                    # 检查并添加点和标签提示
                    if 'points' in obj_prompts and obj_prompts['points']:
                        point_coords = np.array(obj_prompts['points'])
                        point_labels = np.array(obj_prompts['labels'])
                        
                        # 确保点和标签数量匹配, 确保每个point的长度为2
                        if point_coords.shape[0] == point_labels.shape[0] and point_coords.shape[1] == 2 and len(point_labels.shape) == 1:
                            sam_prompt_dict['point_coords'] = point_coords
                            sam_prompt_dict['point_labels'] = point_labels
                        else:
                            print(f"Warning: Mismatch between points ({point_coords.shape[0]}) and labels ({point_labels.shape[0]}). Skipping points for this object.")
                except Exception as e:
                    pass
                
                if sam_prompt_dict:
                    sam_prompt_list_for_image.append(sam_prompt_dict)

            prompts_for_batch.append(sam_prompt_list_for_image)
            
            
        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)
        
        # --- 后续的推理流程保持不变 ---

        # 使用状态管理器，在计算时自动将模型加载到GPU，计算后卸载
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",
            # 推理只需要加载模型参数
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            # 1. 将输入数据移动到CUDA设备
            data = data.to("cuda")

            # 2. 调用策略对象执行核心推理逻辑
            #    您的SAMStrategy中的 'segment' 方法现在会收到我们精心构建的 prompts_for_batch
            output_batch = self.strategy.segment(batch=data)
            
            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list

            # 4. 将输入和输出数据移回CPU，释放GPU显存
            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data
    
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_samr1(self, data: "DataProto") -> "DataProto":
        """
        执行SAM模型的分割推理任务。
        此函数已更新，可以正确解析LLM输出并构建包含多种提示的SAM输入。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'responses' 等
                            SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}
        # 从data.batch中解码LLM的文本响应
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        prompts_for_batch = []
        # breakpoint()
        for response in response_text_list:
            # 1. 解析单个响应字符串，获取所有物体的提示信息
            # parsed_objects 的例子: [{'box': [x,y,x,y], 'points': [[x,y]], 'labels': [1]}, ...]
            parsed_objects = parse_visual_prompt_from_content_samr1(response)
            
            sam_prompt_list_for_image = []
            # 2. 遍历这张图片中所有被识别出的物体
            for obj_prompts in parsed_objects:

                sam_prompt_dict = {}
                try:
                    # 检查并添加边界框提示
                    if 'box' in obj_prompts and obj_prompts['box']:
                        # SAM 通常期望 box 是 (1, 4) 的 numpy 数组
                        if len(obj_prompts['box']) == 4:
                            sam_prompt_dict['box'] = np.array(obj_prompts['box'])

                    # 检查并添加点和标签提示
                    if 'points' in obj_prompts and obj_prompts['points']:
                        point_coords = np.array(obj_prompts['points'])
                        point_labels = np.array(obj_prompts['labels'])
                        
                        # 确保点和标签数量匹配, 确保每个point的长度为2
                        if point_coords.shape[0] == point_labels.shape[0] and point_coords.shape[1] == 2 and len(point_labels.shape) == 1:
                            sam_prompt_dict['point_coords'] = point_coords
                            sam_prompt_dict['point_labels'] = point_labels
                        else:
                            print(f"Warning: Mismatch between points ({point_coords.shape[0]}) and labels ({point_labels.shape[0]}). Skipping points for this object.")
                except Exception as e:
                    pass
                
                if sam_prompt_dict:
                    sam_prompt_list_for_image.append(sam_prompt_dict)

            prompts_for_batch.append(sam_prompt_list_for_image)
            
            
        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)
        
        # --- 后续的推理流程保持不变 ---

        # 使用状态管理器，在计算时自动将模型加载到GPU，计算后卸载
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",
            # 推理只需要加载模型参数
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            # 1. 将输入数据移动到CUDA设备
            data = data.to("cuda")

            # 2. 调用策略对象执行核心推理逻辑
            #    您的SAMStrategy中的 'segment' 方法现在会收到我们精心构建的 prompts_for_batch
            output_batch = self.strategy.segment(batch=data)
            
            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list

            # 4. 将输入和输出数据移回CPU，释放GPU显存
            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data
