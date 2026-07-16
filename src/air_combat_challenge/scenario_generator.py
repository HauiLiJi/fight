import json
import random

class ScenarioGenerator:
    """
    随机想定生成器
    支持随机单位数量、单位类型、初始位置
    """
    
    def __init__(self, config):
        """
        初始化随机想定生成器
        
        Args:
            config: 配置字典，包含随机想定的参数
        """
        self.random_config = config
    
    def generate(self):
        """
        生成随机想定
        
        Returns:
            dict: 想定数据
        """
        scenario_data = {
            'aircrafts': {}
        }
        
        # 为红蓝双方生成单位
        for side in ['red', 'blue']:
            # 随机单位数量
            min_count = self.random_config['min_aircrafts_per_side']
            max_count = self.random_config['max_aircrafts_per_side']
            aircraft_count = random.randint(min_count, max_count)
            
            # 生成单位
            for i in range(1, aircraft_count + 1):
                aircraft_id = f'{side}_fighter_{i}'
                aircraft_type = random.choice(self.random_config['aircraft_types'])
                
                # 随机位置
                pos_range = self.random_config['position_range'][side]
                latitude = random.uniform(pos_range['latitude'][0], pos_range['latitude'][1])
                longitude = random.uniform(pos_range['longitude'][0], pos_range['longitude'][1])
                altitude = random.uniform(pos_range['altitude'][0], pos_range['altitude'][1])
                heading = random.uniform(pos_range['heading'][0], pos_range['heading'][1])
                
                # 构建单位信息
                scenario_data['aircrafts'][aircraft_id] = {
                    'plat_id': aircraft_id,
                    'plat_type': aircraft_type,
                    'side': side,
                    'lat': latitude,
                    'lon': longitude,
                    'alt': altitude,
                    'heading': heading,
                    'pitch': 0.0,
                    'roll': 0.0,
                    'speed': 500.0  # 默认速度
                }
        return scenario_data
    
    def generate_to_file(self, output_path):
        """
        生成随机想定并保存到文件
        
        Args:
            output_path: 输出文件路径
        """
        scenario_data = self.generate()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_data, f, indent=2, ensure_ascii=False)
        return scenario_data
