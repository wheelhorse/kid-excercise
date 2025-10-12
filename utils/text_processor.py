"""
Text processing utilities for Chinese and English text
"""
import re
import jieba
from typing import List, Set, Optional
from zhconv import convert
from utils.logger import Logger

logger = Logger.get_logger("hybrid_search.text_processor")


class TextProcessor:
    """Text processing utility for bilingual content"""
    
    def __init__(self):
        """Initialize text processor"""
        # Load jieba dictionary for better Chinese segmentation
        jieba.initialize()
        
        # Common English stop words
        self.english_stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their'
        }
        
        # Common Chinese stop words
        self.chinese_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '那', '里', '就是', '还', '把', '被', '让',
            '给', '从', '向', '往', '跟', '同', '对', '为了', '因为', '所以', '如果',
            '虽然', '但是', '然而', '而且', '或者', '否则', '不过', '可是', '只是'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert traditional Chinese to simplified
        text = convert(text, 'zh-cn')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Chinese characters, letters, numbers
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:()\-\[\]{}"]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_names(self, text):
        # Simple rule: 2–3 consecutive Chinese characters, maybe a surname from common family names
        #surnames = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜谢邹喻"
        surnames = (
            "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
            "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史"
            "唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟"
            "平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项"
            "祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田"
            "樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左"
            "石崔吉龚程嵇邢滑裴陆荣翁荀羊於惠甄魏加封芮羿储靳汲邴糜松井段富"
            "巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武"
            "符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池"
            "乔阴鬱胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍却璩桑桂濮牛寿通"
            "边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾"
            "终暨居衡步都耿满弘匡国文寇广禄阙东殴殳沃利蔚越夔隆师巩厍聂晁勾"
            "敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖"
            "益桓公万俟司马上官欧阳夏侯诸葛闻人东方赫连皇甫尉迟公羊澹台公冶"
            "宗政濮阳淳于单于太叔申屠公孙仲孙轩辕令狐锺离宇文长孙慕容鲜于闾丘"
            "司徒司空亓官司寇子车颛孙端木巫马公西漆雕乐正壤驷公良拓跋夹谷宰父"
            "谷梁段干百里东郭南门呼延归海羊舌微生梁丘左丘东门西门商牟佘佴伯赏"
            "南宫墨哈谯笪年爱阳佟"
        )

        compound_surnames = [
            "欧阳","司马","诸葛","上官","东方","夏侯","皇甫","尉迟","公羊","澹台",
            "公冶","宗政","濮阳","淳于","单于","太叔","申屠","公孙","仲孙","轩辕",
            "令狐","锺离","宇文","长孙","慕容","鲜于","闾丘","司徒","司空","亓官",
            "司寇","子车","颛孙","端木","巫马","公西","漆雕","乐正","壤驷","公良",
            "拓跋","夹谷","宰父","谷梁","段干","百里","东郭","南门","呼延","归海",
            "羊舌","微生","梁丘","左丘","东门","西门","商牟","佘佴","伯赏","南宫"
        ]
        
        pattern = re.compile("(?:" + "|".join(compound_surnames) + "|" + f"[{surnames}])[\u4e00-\u9fa5]{{1,2}}")

        return pattern.findall(text)

    def tokenize_chinese(self, text: str) -> List[str]:
        """Tokenize Chinese text using jieba.cut_for_search for optimal search performance"""
        if not text:
            return []
        
        all_tokens = []

        all_tokens.extend(self.detect_names(text))
        
        # Use jieba.cut_for_search for better search tokenization
        # This automatically generates more granular tokens for search scenarios
        search_tokens = list(jieba.cut_for_search(text))
        
        # Filter out stop words and short tokens
        for token in search_tokens:
            token = token.strip()
            if (len(token) >= 1 and 
                token not in self.chinese_stop_words and
                not token.isspace()):
                all_tokens.append(token)
        
        ## Add character-level subsequences for Chinese text (especially names)
        ## This enables partial matching like '徐佳' finding '徐佳芸'
        ## Since this function handles Chinese text, we can work directly with the characters
        #if text and any('\u4e00' <= c <= '\u9fff' for c in text):
        #    # Generate character-level n-grams (1-4 characters) from the text directly
        #    for i in range(len(text)):
        #        char = text[i]
        #        # Skip non-Chinese characters and stop words
        #        if '\u4e00' <= char <= '\u9fff' and char not in self.chinese_stop_words:
        #            # Single characters (for very short queries)
        #            all_tokens.append(char)
        #            
        #            # 2-character combinations
        #            if i < len(text) - 1 and '\u4e00' <= text[i + 1] <= '\u9fff':
        #                bigram = text[i:i + 2]
        #                if bigram not in self.chinese_stop_words:
        #                    all_tokens.append(bigram)
        #            
        #            # 3-character combinations
        #            if i < len(text) - 2 and all('\u4e00' <= text[i + j] <= '\u9fff' for j in range(3)):
        #                trigram = text[i:i + 3]
        #                all_tokens.append(trigram)
        #            
        #            # 4-character combinations
        #            if i < len(text) - 3 and all('\u4e00' <= text[i + j] <= '\u9fff' for j in range(4)):
        #                four_gram = text[i:i + 4]
        #                all_tokens.append(four_gram)
        
        return all_tokens
    
    def tokenize_english(self, text: str) -> List[str]:
        """Tokenize English text"""
        if not text:
            return []
        
        # Split by word boundaries and clean
        tokens = re.findall(r'\b[a-zA-Z0-9+#.]{2,}\b', text.lower())
        
        # Filter out stop words
        filtered_tokens = [
            token for token in tokens 
            if token not in self.english_stop_words and len(token) >= 2
        ]
        
        return filtered_tokens
    
    def tokenize_mixed(self, text: str) -> List[str]:
        """Tokenize mixed Chinese and English text"""
        if not text:
            return []
        
        # Clean text first
        text = self.clean_text(text)
        
        # Separate Chinese and English parts
        chinese_parts = re.findall(r'[\u4e00-\u9fff]+', text)
        english_parts = re.findall(r'[a-zA-Z0-9+#.]+', text)
        
        all_tokens = []
        
        # Process Chinese parts
        for part in chinese_parts:
            all_tokens.extend(self.tokenize_chinese(part))
        
        # Process English parts
        english_text = ' '.join(english_parts)
        all_tokens.extend(self.tokenize_english(english_text))
        
        return list(set(all_tokens))  # Remove duplicates
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract technical skills and keywords"""
        if not text:
            return []
        
        # Technical skill patterns
        skill_patterns = [
            r'\b(?:python|java|javascript|typescript|c\+\+|c#|php|ruby|go|rust|swift|kotlin)\b',
            r'\b(?:react|vue|angular|django|flask|spring|express|laravel)\b',
            r'\b(?:mysql|postgresql|mongodb|redis|elasticsearch|oracle)\b',
            r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins|git)\b',
            r'\b(?:machine learning|deep learning|ai|nlp|computer vision)\b',
            r'\b(?:pandas|numpy|tensorflow|pytorch|scikit-learn)\b'
        ]
        
        skills = []
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            skills.extend(matches)
        
        # Add Chinese technical terms
        chinese_skills = re.findall(r'(?:机器学习|深度学习|人工智能|数据科学|算法|编程|开发|软件工程)', text)
        skills.extend(chinese_skills)
        
        return list(set(skills))
    
    def create_search_text(self, candidate_data: dict) -> str:
        """Create searchable text from candidate data"""
        parts = []
        
        # Add name
        if candidate_data.get('first_name'):
            parts.append(candidate_data['first_name'])
        if candidate_data.get('last_name'):
            parts.append(candidate_data['last_name'])
        
        # Add key skills
        if candidate_data.get('key_skills'):
            parts.append(candidate_data['key_skills'])
        
        # Add notes
        if candidate_data.get('notes'):
            parts.append(candidate_data['notes'])
        
        # Add resume text
        if candidate_data.get('resume_text'):
            parts.append(candidate_data['resume_text'])
        
        # Join and clean
        search_text = ' '.join(parts)
        return self.clean_text(search_text)
    
    def create_bm25_tokens(self, text: str) -> List[str]:
        """Create tokens for BM25 sparse embedding"""
        if not text:
            return []
        
        # Get mixed language tokens
        tokens = self.tokenize_mixed(text)
        
        # Add bigrams for better matching
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            bigrams.append(bigram)
        
        # Combine unigrams and bigrams
        all_tokens = tokens + bigrams
        
        # Filter by length and frequency
        filtered_tokens = [
            token for token in all_tokens 
            if 1 <= len(token) <= 50
        ]
        
        return filtered_tokens
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple token-based similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        tokens1 = set(self.tokenize_mixed(text1))
        tokens2 = set(self.tokenize_mixed(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def highlight_matches(self, text: str, query_tokens: List[str], max_length: int = 300) -> str:
        """Highlight matching tokens in text snippet"""
        if not text or not query_tokens:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Find first match position
        text_lower = text.lower()
        first_match_pos = len(text)
        
        for token in query_tokens:
            pos = text_lower.find(token.lower())
            if pos >= 0:
                first_match_pos = min(first_match_pos, pos)
        
        # Extract snippet around first match
        start = max(0, first_match_pos - 100)
        end = min(len(text), start + max_length)
        snippet = text[start:end]
        
        # Simple highlighting (for terminal display)
        for token in query_tokens:
            if len(token) >= 2:
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                snippet = pattern.sub(f"**{token.upper()}**", snippet)
        
        return snippet


# Global text processor instance
text_processor = TextProcessor()
