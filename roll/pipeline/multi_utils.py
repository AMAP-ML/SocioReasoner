import re
import ast

def parse_points_text_from_content(content):
    """
    """
    # 正则表达式部分无需修改，它能捕获标签内的任何内容
    answer_pattern = r"<answer>(.*?)</answer>"
    
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:
        points_text = answer_match.group(1)
        return points_text.strip()
    else:
        # print("未找到 <answer> 标签")
        return ""