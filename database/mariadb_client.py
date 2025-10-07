"""
MariaDB dummy wrapper for the Resume Retrieval System
This is a dummy implementation that simulates database operations
"""
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from utils.logger import Logger, log_method
from config import MARIADB_CONFIG

logger = Logger.get_logger("hybrid_search.mariadb")


@dataclass
class CandidateRecord:
    """Data class for candidate records"""
    candidate_id: int
    first_name: str
    last_name: str
    email1: str
    key_skills: str
    notes: str
    date_modified: datetime
    resume_text: Optional[str] = None
    attachment_id: Optional[int] = None


class MariaDBClient:
    """Dummy MariaDB client that simulates real database operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the dummy MariaDB client"""
        self.config = config or MARIADB_CONFIG
        self.connected = False
        self.last_sync_time = None
        
        # Generate dummy data for testing
        self._dummy_data = self._generate_dummy_data()
        logger.info("MariaDB dummy client initialized")
    
    @log_method("hybrid_search.mariadb")
    def connect(self) -> bool:
        """Simulate database connection"""
        try:
            # Simulate connection delay
            time.sleep(0.1)
            self.connected = True
            logger.info("Connected to MariaDB (dummy mode)")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MariaDB: {str(e)}")
            return False
    
    @log_method("hybrid_search.mariadb")
    def disconnect(self):
        """Simulate database disconnection"""
        self.connected = False
        logger.info("Disconnected from MariaDB")
    
    def _generate_dummy_data(self) -> List[CandidateRecord]:
        """Generate dummy candidate data for testing"""
        candidates = []
        
        # Sample candidate data with mixed Chinese and English
        sample_data = [
            {
                "first_name": "张", "last_name": "伟",
                "email": "zhang.wei@email.com",
                "key_skills": "Python, Django, MySQL, 机器学习, 数据分析",
                "notes": "5年Python开发经验，擅长Web开发和数据分析，熟悉机器学习算法",
                "resume_text": "张伟，高级Python开发工程师，5年互联网开发经验。精通Python、Django、Flask框架，熟悉MySQL、PostgreSQL数据库。具有丰富的机器学习项目经验，使用过TensorFlow、PyTorch等深度学习框架。参与过多个大型Web应用的开发和维护。"
            },
            {
                "first_name": "John", "last_name": "Smith",
                "email": "john.smith@email.com",
                "key_skills": "Java, Spring Boot, Microservices, AWS, Docker",
                "notes": "Senior Java developer with 8 years of experience in enterprise applications",
                "resume_text": "John Smith - Senior Java Developer. 8+ years of experience in enterprise software development. Expert in Java, Spring Boot, Spring Cloud, and microservices architecture. Extensive experience with AWS cloud services, Docker containerization, and Kubernetes orchestration. Led multiple teams in agile development environments."
            },
            {
                "first_name": "李", "last_name": "小明",
                "email": "li.xiaoming@email.com",
                "key_skills": "React, Vue.js, Node.js, TypeScript, 前端开发",
                "notes": "前端开发专家，精通现代前端框架和工具链",
                "resume_text": "李小明，前端开发工程师，6年前端开发经验。精通React、Vue.js、Angular等现代前端框架，熟悉TypeScript、Webpack、Vite等工具。有丰富的移动端H5开发经验，熟悉微信小程序开发。参与过多个大型电商平台的前端架构设计。"
            },
            {
                "first_name": "Sarah", "last_name": "Johnson",
                "email": "sarah.johnson@email.com",
                "key_skills": "Data Science, Python, Machine Learning, SQL, Tableau",
                "notes": "Data scientist with expertise in ML algorithms and statistical analysis",
                "resume_text": "Sarah Johnson - Data Scientist with 4 years of experience in machine learning and statistical analysis. Proficient in Python, R, SQL, and various ML libraries including scikit-learn, pandas, and numpy. Experience with big data technologies like Spark and Hadoop. Strong background in data visualization using Tableau and matplotlib."
            },
            {
                "first_name": "王", "last_name": "芳",
                "email": "wang.fang@email.com",
                "key_skills": "iOS, Swift, Objective-C, 移动开发, Xcode",
                "notes": "iOS开发工程师，专注移动应用开发",
                "resume_text": "王芳，iOS开发工程师，7年移动应用开发经验。精通Swift、Objective-C语言，熟悉iOS SDK和各种第三方框架。有丰富的App Store上架经验，参与过多款千万级用户的移动应用开发。熟悉MVVM、MVP等架构模式。"
            },
            {
                "first_name": "Michael", "last_name": "Chen",
                "email": "michael.chen@email.com",
                "key_skills": "DevOps, Kubernetes, Jenkins, Terraform, Linux",
                "notes": "DevOps engineer with cloud infrastructure expertise",
                "resume_text": "Michael Chen - DevOps Engineer with 6 years of experience in cloud infrastructure and automation. Expert in Kubernetes, Docker, Jenkins, and Terraform. Strong background in Linux system administration and shell scripting. Experience with AWS, Azure, and GCP cloud platforms. Implemented CI/CD pipelines for multiple organizations."
            },
            {
                "first_name": "陈", "last_name": "佳",
                "email": "chen.jia@email.com",
                "key_skills": "算法工程师, Python, C++, 深度学习, NLP",
                "notes": "算法工程师，专注自然语言处理和深度学习",
                "resume_text": "陈佳，算法工程师，5年人工智能算法研发经验。精通Python、C++编程，熟悉TensorFlow、PyTorch深度学习框架。专注于自然语言处理、计算机视觉等AI领域。发表过多篇顶级会议论文，具有丰富的工业界AI项目落地经验。"
            },
            {
                "first_name": "Emma", "last_name": "Wilson",
                "email": "emma.wilson@email.com",
                "key_skills": "Product Management, Agile, Scrum, Analytics, UX/UI",
                "notes": "Senior product manager with 10 years of experience in tech products",
                "resume_text": "Emma Wilson - Senior Product Manager with 10+ years of experience in technology product development. Expert in agile methodologies, user experience design, and data-driven product decisions. Led cross-functional teams to deliver successful products in fintech and e-commerce sectors. Strong analytical skills with experience in A/B testing and user research."
            }
        ]
        
        # Generate candidates with random timestamps
        for i, data in enumerate(sample_data):
            # Random modification time within last 30 days
            days_ago = random.randint(0, 30)
            mod_time = datetime.now() - timedelta(days=days_ago, 
                                                hours=random.randint(0, 23),
                                                minutes=random.randint(0, 59))
            
            candidate = CandidateRecord(
                candidate_id=i + 1,
                first_name=data["first_name"],
                last_name=data["last_name"],
                email1=data["email"],
                key_skills=data["key_skills"],
                notes=data["notes"],
                date_modified=mod_time,
                resume_text=data["resume_text"],
                attachment_id=i + 100  # Dummy attachment ID
            )
            candidates.append(candidate)
        
        return candidates
    
    @log_method("hybrid_search.mariadb")
    def get_candidates_with_resumes(self, limit: Optional[int] = None) -> List[CandidateRecord]:
        """Get all candidates with resume attachments"""
        if not self.connected:
            logger.error("Not connected to database")
            return []
        
        # Simulate query delay
        time.sleep(0.2)
        
        candidates = self._dummy_data.copy()
        
        if limit:
            candidates = candidates[:limit]
        
        logger.info(f"Retrieved {len(candidates)} candidate records")
        return candidates
    
    @log_method("hybrid_search.mariadb")
    def get_candidates_modified_after(self, timestamp: datetime, limit: Optional[int] = None) -> List[CandidateRecord]:
        """Get candidates modified after a specific timestamp"""
        if not self.connected:
            logger.error("Not connected to database")
            return []
        
        # Simulate query delay
        time.sleep(0.1)
        
        modified_candidates = [
            candidate for candidate in self._dummy_data
            if candidate.date_modified > timestamp
        ]
        
        if limit:
            modified_candidates = modified_candidates[:limit]
        
        logger.info(f"Retrieved {len(modified_candidates)} candidates modified after {timestamp}")
        return modified_candidates
    
    @log_method("hybrid_search.mariadb")
    def get_candidate_by_id(self, candidate_id: int) -> Optional[CandidateRecord]:
        """Get a specific candidate by ID"""
        if not self.connected:
            logger.error("Not connected to database")
            return None
        
        # Simulate query delay
        time.sleep(0.05)
        
        for candidate in self._dummy_data:
            if candidate.candidate_id == candidate_id:
                logger.info(f"Retrieved candidate {candidate_id}")
                return candidate
        
        logger.warning(f"Candidate {candidate_id} not found")
        return None
    
    @log_method("hybrid_search.mariadb")
    def get_total_candidates_count(self) -> int:
        """Get total number of candidates with resumes"""
        if not self.connected:
            logger.error("Not connected to database")
            return 0
        
        count = len(self._dummy_data)
        logger.info(f"Total candidates count: {count}")
        return count
    
    @log_method("hybrid_search.mariadb")
    def get_last_modification_time(self) -> Optional[datetime]:
        """Get the latest modification timestamp"""
        if not self.connected or not self._dummy_data:
            return None
        
        latest = max(candidate.date_modified for candidate in self._dummy_data)
        logger.info(f"Latest modification time: {latest}")
        return latest
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            if not self.connected:
                self.connect()
            
            # Simulate a simple query
            time.sleep(0.1)
            count = self.get_total_candidates_count()
            return count >= 0
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def set_last_sync_time(self, timestamp: datetime):
        """Set the last successful sync timestamp"""
        self.last_sync_time = timestamp
        logger.info(f"Set last sync time to: {timestamp}")
    
    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the last successful sync timestamp"""
        return self.last_sync_time
    
    def simulate_data_update(self):
        """Simulate new data updates (for testing sync functionality)"""
        if not self._dummy_data:
            return
        
        # Randomly update some records
        num_updates = random.randint(1, 3)
        updated_records = random.sample(self._dummy_data, min(num_updates, len(self._dummy_data)))
        
        for record in updated_records:
            record.date_modified = datetime.now()
            record.notes = f"{record.notes} (Updated {datetime.now().strftime('%Y-%m-%d %H:%M')})"
        
        logger.info(f"Simulated updates to {num_updates} records")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Global instance for easy access
mariadb_client = MariaDBClient()
