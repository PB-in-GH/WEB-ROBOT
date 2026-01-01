# experiments/experiment_runner.py
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

class ExperimentRunner:
    """实验运行管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.results = []
        
    def run_experiment(self, 
                      experiment_name: str,
                      audio_file: str,
                      network_trace: str,
                      control_algorithm: str = "lyapunov",
                      duration: int = 30) -> Dict:
        """运行单个实验"""
        
        # 这里调用实际的流式系统
        # 为了演示，我们模拟运行
        
        print(f"运行实验: {experiment_name}")
        print(f"音频: {audio_file}")
        print(f"网络trace: {network_trace}")
        print(f"控制算法: {control_algorithm}")
        
        # 模拟运行结果
        result = {
            'experiment_name': experiment_name,
            'audio_file': audio_file,
            'network_trace': network_trace,
            'control_algorithm': control_algorithm,
            'duration': duration,
            'metrics': {
                'avg_network_score': 85.3,
                'avg_fps': 28.7,
                'buffer_underruns': 3,
                'avg_delay_ms': 45.2,
                'packet_loss_rate': 0.015
            },
            'timeline': self._generate_mock_timeline(duration)
        }
        
        self.results.append(result)
        return result
    
    def _generate_mock_timeline(self, duration: int) -> List[Dict]:
        """生成模拟时间线数据"""
        timeline = []
        for t in range(duration):
            timeline.append({
                'timestamp': t,
                'buffer_size': max(5, min(25, 15 + 5 * (t % 10 - 5))),
                'fps': max(20, min(40, 30 + 5 * (t % 6 - 3))),
                'network_score': max(70, min(95, 85 + 10 * (t % 8 - 4))),
                'packet_loss': max(0, min(0.05, 0.02 + 0.015 * (t % 5 - 2.5)))
            })
        return timeline
    
    def save_results(self, output_file: str):
        """保存实验结果"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"结果已保存到: {output_file}")
        
    def generate_report(self, output_dir: str):
        """生成实验报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建DataFrame
        df_data = []
        for result in self.results:
            row = {
                '实验名称': result['experiment_name'],
                '控制算法': result['control_algorithm'],
                '平均Network Score': result['metrics']['avg_network_score'],
                '平均帧率': result['metrics']['avg_fps'],
                '缓冲区下溢': result['metrics']['buffer_underruns'],
                '平均延迟(ms)': result['metrics']['avg_delay_ms'],
                '丢包率': result['metrics']['packet_loss_rate']
            }
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        
        # 保存CSV
        csv_path = output_path / "experiment_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 生成图表
        self._plot_results(df, output_path)
        
        print(f"报告已生成到: {output_dir}")
        
    def _plot_results(self, df: pd.DataFrame, output_path: Path):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Network Score比较
        ax = axes[0, 0]
        df.plot.bar(x='实验名称', y='平均Network Score', ax=ax, legend=False)
        ax.set_title('平均Network Score比较')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. 帧率比较
        ax = axes[0, 1]
        df.plot.bar(x='实验名称', y='平均帧率', ax=ax, legend=False, color='orange')
        ax.set_title('平均帧率比较')
        ax.set_ylabel('FPS')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. 延迟比较
        ax = axes[1, 0]
        df.plot.bar(x='实验名称', y='平均延迟(ms)', ax=ax, legend=False, color='green')
        ax.set_title('平均延迟比较')
        ax.set_ylabel('延迟 (ms)')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. 综合指标散点图
        ax = axes[1, 1]
        scatter = ax.scatter(df['平均Network Score'], 
                            df['平均帧率'],
                            s=df['缓冲区下溢']*50 + 100,
                            c=df['平均延迟(ms)'],
                            cmap='viridis',
                            alpha=0.6)
        
        ax.set_xlabel('Network Score')
        ax.set_ylabel('平均帧率 (FPS)')
        ax.set_title('综合指标分布')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='平均延迟 (ms)')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        chart_path = output_path / "experiment_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="AI数字人实时传输实验运行器")
    parser.add_argument("--experiments", type=str, required=True,
                       help="实验配置文件路径")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 加载实验配置
    with open(args.experiments, 'r') as f:
        experiments = json.load(f)
    
    runner = ExperimentRunner()
    
    # 运行所有实验
    for exp in experiments:
        runner.run_experiment(
            experiment_name=exp['name'],
            audio_file=exp['audio'],
            network_trace=exp['network_trace'],
            control_algorithm=exp.get('control_algorithm', 'lyapunov'),
            duration=exp.get('duration', 30)
        )
    
    # 保存结果和报告
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    runner.save_results(output_dir / "raw_results.json")
    runner.generate_report(output_dir)
    
    print("所有实验完成！")


if __name__ == "__main__":
    main()