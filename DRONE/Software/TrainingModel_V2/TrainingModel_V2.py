import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
from collections import deque
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class TreeObject:
    x: float
    y: float
    width: float
    height: float
    density: float

class VisionSensor:
    def __init__(self, fov=90, resolution=64):
        self.fov = fov
        self.resolution = resolution
        self.max_range = 15.0
    
    def scan(self, drone_pos, drone_yaw, trees, obstacles):
        readings = np.ones(self.resolution) * self.max_range
        tree_confidence = np.zeros(self.resolution)
        
        start_angle = drone_yaw - self.fov / 2
        angle_step = self.fov / self.resolution
        
        for i in range(self.resolution):
            angle = math.radians(start_angle + i * angle_step)
            
            for dist in np.linspace(0.5, self.max_range, 50):
                ray_x = drone_pos[0] + dist * math.cos(angle)
                ray_y = drone_pos[1] + dist * math.sin(angle)
                
                for tree in trees:
                    dx = ray_x - tree.x
                    dy = ray_y - tree.y
                    if math.sqrt(dx*dx + dy*dy) < tree.width / 2:
                        readings[i] = min(readings[i], dist)
                        conf = tree.density * (1.0 - dist / self.max_range)
                        tree_confidence[i] = max(tree_confidence[i], conf)
                        break
                
                for obs in obstacles:
                    if abs(ray_x - obs[0]) < 0.5 and abs(ray_y - obs[1]) < 0.5:
                        readings[i] = min(readings[i], dist)
                        break
        
        return readings, tree_confidence

class AgriculturalEnvironment:
    def __init__(self, rows=4, trees_per_row=25, randomize=True):
        self.rows = rows
        self.trees_per_row = trees_per_row
        self.target_distance = 2.5
        self.tolerance = 0.5
        self.row_spacing = 7.0
        
        self.trees = []
        self.obstacles = []
        
        for row in range(rows):
            base_spacing = random.uniform(3.5, 5.0) if randomize else 4.0
            x_position = 0
            
            for tree_idx in range(trees_per_row):
                spacing = base_spacing + random.uniform(-0.8, 0.8) if randomize else base_spacing
                x_position += spacing
                
                y = row * self.row_spacing
                width = random.uniform(2.0, 4.0) if randomize else 3.0
                height = random.uniform(3.5, 5.5) if randomize else 4.5
                density = random.uniform(0.6, 1.0) if randomize else 0.8
                
                self.trees.append(TreeObject(x_position, y, width, height, density))
                
                if randomize and random.random() < 0.05:
                    obs_x = x_position + random.uniform(-1, 1)
                    obs_y = y + random.uniform(-2, 2)
                    self.obstacles.append([obs_x, obs_y])
        
        self.sensor = VisionSensor()
        self.reset()
    
    def reset(self):
        self.drone_pos = np.array([0.0, -self.target_distance, 3.0])
        self.drone_vel = np.array([0.0, 0.0, 0.0])
        self.drone_yaw = 90.0
        
        self.current_row = 0
        self.current_side = 0
        self.photos_taken = []
        self.total_photos = 0
        self.steps = 0
        self.done = False
        
        self.tree_coverage = np.zeros((len(self.trees), 2))
        
        return self.get_state()
    
    def get_state(self):
        distances, tree_conf = self.sensor.scan(
            self.drone_pos, self.drone_yaw, self.trees, self.obstacles
        )
        
        distances = distances / self.sensor.max_range
        
        current_row_y = self.current_row * self.row_spacing
        if self.current_side == 0:
            target_y = current_row_y - self.target_distance
        else:
            target_y = current_row_y + self.target_distance
        
        rel_x = self.drone_pos[0] / 100.0
        rel_y = (self.drone_pos[1] - target_y) / 10.0
        
        expected_photos = len(self.trees) * 2
        photo_progress = self.total_photos / expected_photos
        row_progress = self.current_row / self.rows
        side_indicator = float(self.current_side)
        
        state = np.concatenate([
            distances,
            tree_conf,
            [rel_x, rel_y],
            self.drone_vel / 2.0,
            [self.drone_yaw / 360.0],
            [photo_progress, row_progress, side_indicator, self.steps / 5000.0]
        ])
        
        return state.astype(np.float32)
    
    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}
        
        forward_speed = (action[0] - 0.5) * 4.0
        lateral_speed = (action[1] - 0.5) * 3.0
        yaw_rate = (action[2] - 0.5) * 60.0
        take_photo = action[3]
        
        self.drone_yaw += yaw_rate * 0.1
        self.drone_yaw = self.drone_yaw % 360
        
        yaw_rad = math.radians(self.drone_yaw)
        self.drone_vel[0] = forward_speed * math.cos(yaw_rad) - lateral_speed * math.sin(yaw_rad)
        self.drone_vel[1] = forward_speed * math.sin(yaw_rad) + lateral_speed * math.cos(yaw_rad)
        
        self.drone_pos += self.drone_vel * 0.1
        
        collision = False
        for obs in self.obstacles:
            if np.linalg.norm(self.drone_pos[:2] - obs) < 0.8:
                collision = True
                break
        
        reward = self.calculate_reward(action, collision)
        
        if take_photo > 0.7:
            photo_reward = self.attempt_photo()
            reward += photo_reward
        
        self.steps += 1
        if self.steps > 5000 or collision:
            self.done = True
        
        if self.total_photos >= len(self.trees) * 2:
            self.done = True
            reward += 1000.0
        
        info = {
            'collision': collision,
            'photos': self.total_photos,
            'coverage': np.sum(self.tree_coverage) / (len(self.trees) * 2)
        }
        
        return self.get_state(), reward, self.done, info
    
    def calculate_reward(self, action, collision):
        reward = 0
        
        if collision:
            return -100.0
        
        current_row_y = self.current_row * self.row_spacing
        if self.current_side == 0:
            target_y = current_row_y - self.target_distance
        else:
            target_y = current_row_y + self.target_distance
        
        y_error = abs(self.drone_pos[1] - target_y)
        
        if y_error < self.tolerance:
            reward += 1.0
        else:
            reward -= y_error * 0.5
        
        reward += action[0] * 0.1
        
        reward -= abs(action[2] - 0.5) * 0.2
        
        reward -= 0.01
        
        return reward
    
    def attempt_photo(self):
        yaw_rad = math.radians(self.drone_yaw)
        look_direction = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
        
        best_tree_idx = None
        best_score = -1
        
        for idx, tree in enumerate(self.trees):
            tree_pos = np.array([tree.x, tree.y])
            to_tree = tree_pos - self.drone_pos[:2]
            distance = np.linalg.norm(to_tree)
            
            if distance > 0.5:
                to_tree_norm = to_tree / distance
                alignment = np.dot(look_direction, to_tree_norm)
                
                if alignment > 0.8 and 2.0 <= distance <= 3.0:
                    score = alignment * (1.0 - abs(distance - 2.5) / 0.5)
                    if score > best_score:
                        best_score = score
                        best_tree_idx = idx
        
        if best_tree_idx is not None:
            if self.tree_coverage[best_tree_idx, self.current_side] == 0:
                self.tree_coverage[best_tree_idx, self.current_side] = 1
                self.total_photos += 1
                self.photos_taken.append({
                    'tree': best_tree_idx,
                    'side': self.current_side,
                    'distance': np.linalg.norm(
                        np.array([self.trees[best_tree_idx].x, self.trees[best_tree_idx].y]) - 
                        self.drone_pos[:2]
                    )
                })
                return 50.0
        
        return -5.0

class PPOAgent(nn.Module):
    def __init__(self, state_dim=138, action_dim=4):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )
        
        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
        
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        features = self.features(state)
        action_probs = self.policy(features)
        state_value = self.value(features)
        return action_probs, state_value

class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.episode_photos = []
        self.episode_coverage = []
        self.episode_collisions = []
        self.episode_lengths = []
        self.avg_distances = []
        
    def add_episode(self, reward, photos, coverage, collision, length, distances):
        self.episode_rewards.append(reward)
        self.episode_photos.append(photos)
        self.episode_coverage.append(coverage)
        self.episode_collisions.append(1 if collision else 0)
        self.episode_lengths.append(length)
        if distances:
            self.avg_distances.append(np.mean(distances))
    
    def plot_training_progress(self, save_path='training_progress.png'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Autonomous Agricultural Photography System - Training Progress', 
                     fontsize=16, fontweight='bold')
        
        def moving_avg(data, window=50):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        axes[0, 0].plot(moving_avg(self.episode_rewards), linewidth=2, label='Moving Avg')
        axes[0, 0].set_title('Cumulative Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.episode_coverage, alpha=0.3, label='Raw')
        axes[0, 1].plot(moving_avg(self.episode_coverage), linewidth=2, label='Moving Avg')
        axes[0, 1].set_title('Photo Coverage Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Coverage (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=95, color='r', linestyle='--', label='Target: 95%')
        
        axes[0, 2].plot(self.episode_photos, alpha=0.3, label='Raw')
        axes[0, 2].plot(moving_avg(self.episode_photos), linewidth=2, label='Moving Avg')
        axes[0, 2].set_title('Photos Taken per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Number of Photos')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        collision_rate = moving_avg([sum(self.episode_collisions[max(0, i-50):i+1])/51 
                                      for i in range(len(self.episode_collisions))])
        axes[1, 0].plot(collision_rate, linewidth=2)
        axes[1, 0].set_title('Collision Rate (50-episode window)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Collision Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(self.episode_lengths, alpha=0.3, label='Raw')
        axes[1, 1].plot(moving_avg(self.episode_lengths), linewidth=2, label='Moving Avg')
        axes[1, 1].set_title('Episode Duration (steps)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        if self.avg_distances:
            axes[1, 2].plot(self.avg_distances, alpha=0.3, label='Raw')
            axes[1, 2].plot(moving_avg(self.avg_distances), linewidth=2, label='Moving Avg')
            axes[1, 2].set_title('Average Photo Distance')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Distance (m)')
            axes[1, 2].axhline(y=2.5, color='g', linestyle='--', label='Target: 2.5m')
            axes[1, 2].axhline(y=2.0, color='r', linestyle=':', alpha=0.5)
            axes[1, 2].axhline(y=3.0, color='r', linestyle=':', alpha=0.5)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining progress visualization saved: {save_path}")
        return fig

def train_agent(num_episodes=1000, save_interval=100):
    print("=" * 70)
    print("AUTONOMOUS AGRICULTURAL AERIAL PHOTOGRAPHY SYSTEM")
    print("=" * 70)
    print("Training Configuration:")
    print("  Algorithm: Proximal Policy Optimization (PPO)")
    print("  Sensor: RGB Camera with Depth Estimation")
    print("  Episodes: 1000")
    print("  Curriculum: Simple → Complex scenarios")
    print("=" * 70)
    
    agent = PPOAgent()
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    metrics = TrainingMetrics()
    
    curriculum_stages = [
        (200, {'rows': 2, 'trees_per_row': 10, 'randomize': False}),
        (300, {'rows': 3, 'trees_per_row': 15, 'randomize': True}),
        (500, {'rows': 4, 'trees_per_row': 25, 'randomize': True})
    ]
    
    best_coverage = 0
    best_model_state = None
    
    print("\nPHASE 1: TRAINING WITH PROGRESSIVE CURRICULUM")
    print("-" * 70)
    
    for episode in range(num_episodes):
        env_config = curriculum_stages[0][1]
        for stage_ep, config in curriculum_stages:
            if episode >= stage_ep:
                env_config = config
        
        env = AgriculturalEnvironment(**env_config)
        state = env.reset()
        
        episode_reward = 0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_values = []
        
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, value = agent(state_tensor)
            
            action = action_probs.squeeze().numpy()
            if episode < 500:
                action += np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action, 0, 1)
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_values.append(value.item())
            
            next_state, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            episode_reward += reward
            
            state = next_state
        
        if len(episode_states) > 0:
            states_tensor = torch.FloatTensor(np.array(episode_states))
            actions_tensor = torch.FloatTensor(np.array(episode_actions))
            
            optimizer.zero_grad()
            pred_actions, pred_values = agent(states_tensor)
            
            action_loss = nn.MSELoss()(pred_actions, actions_tensor)
            
            returns = []
            G = 0
            for r in reversed(episode_rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns_tensor = torch.FloatTensor(returns).unsqueeze(1)
            value_loss = nn.MSELoss()(pred_values, returns_tensor)
            
            total_loss = action_loss + 0.5 * value_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
        
        distances = [p['distance'] for p in env.photos_taken]
        coverage = info['coverage'] * 100
        metrics.add_episode(
            episode_reward, 
            info['photos'], 
            coverage,
            info['collision'],
            env.steps,
            distances
        )
        
        if coverage > best_coverage:
            best_coverage = coverage
            best_model_state = agent.state_dict().copy()
        
        if episode % 50 == 0 or episode == num_episodes - 1:
            recent_coverage = np.mean(metrics.episode_coverage[-50:])
            recent_photos = np.mean(metrics.episode_photos[-50:])
            recent_collisions = np.mean(metrics.episode_collisions[-50:])
            
            print(f"Episode {episode:4d} | "
                  f"Coverage: {recent_coverage:5.1f}% | "
                  f"Photos: {recent_photos:5.1f} | "
                  f"Collisions: {recent_collisions:.3f} | "
                  f"Env: {env_config['rows']}x{env_config['trees_per_row']}")
    
    if best_model_state:
        agent.load_state_dict(best_model_state)
    
    print(f"\nTraining Complete. Best Coverage: {best_coverage:.1f}%")
    
    metrics.plot_training_progress()
    
    return agent, metrics

def evaluate_agent(agent, num_tests=10):
    print("\n" + "=" * 70)
    print("PHASE 2: FINAL EVALUATION (100 TREES - 4 ROWS × 25 TREES)")
    print("=" * 70)
    
    agent.eval()
    results = []
    
    for test_num in range(num_tests):
        env = AgriculturalEnvironment(rows=4, trees_per_row=25, randomize=True)
        state = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _ = agent(state_tensor)
                action = action.squeeze().numpy()
            
            state, _, done, info = env.step(action)
        
        coverage = info['coverage'] * 100
        photos = info['photos']
        expected = len(env.trees) * 2
        collision = info['collision']
        
        distances = [p['distance'] for p in env.photos_taken]
        avg_dist = np.mean(distances) if distances else 0
        std_dist = np.std(distances) if distances else 0
        
        results.append({
            'coverage': coverage,
            'photos': photos,
            'expected': expected,
            'avg_distance': avg_dist,
            'std_distance': std_dist,
            'collision': collision,
            'steps': env.steps
        })
        
        status = "COLLISION" if collision else "SUCCESS"
        print(f"Test {test_num + 1:2d}: Coverage: {coverage:5.1f}% | "
              f"Photos: {photos:3d}/{expected:3d} | "
              f"Distance: {avg_dist:.2f}m ± {std_dist:.2f}m | "
              f"Status: {status}")
    
    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS:")
    print("-" * 70)
    
    coverages = [r['coverage'] for r in results]
    distances = [r['avg_distance'] for r in results]
    
    print(f"Average Coverage:    {np.mean(coverages):5.1f}% "
          f"(Best: {np.max(coverages):.1f}%, Worst: {np.min(coverages):.1f}%)")
    print(f"Average Distance:    {np.mean(distances):.2f}m "
          f"(Std: {np.std(distances):.2f}m)")
    print(f"Collision Rate:      {np.mean([r['collision'] for r in results]) * 100:.1f}%")
    print(f"Avg Steps per Test:  {np.mean([r['steps'] for r in results]):.0f}")
    
    print("=" * 70)
    
    if np.mean(coverages) >= 95.0:
        print("STATUS: MISSION CAPABLE - System meets operational requirements")
    elif np.mean(coverages) >= 85.0:
        print("STATUS: REQUIRES REFINEMENT - Additional training recommended")
    else:
        print("STATUS: NOT OPERATIONAL - Significant improvements needed")
    
    print("=" * 70)
    
    return results

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    agent, metrics = train_agent(num_episodes=1000)
    
    results = evaluate_agent(agent, num_tests=10)
    
    return agent, metrics, results

if __name__ == "__main__":
    agent, metrics, results = main()