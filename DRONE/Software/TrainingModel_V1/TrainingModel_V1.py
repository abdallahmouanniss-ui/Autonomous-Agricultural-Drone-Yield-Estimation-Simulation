import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

class DronePhotographyController(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(4, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1)
        )
        
        self.speed_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.photo_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        speed = self.speed_head(features)
        photo = self.photo_head(features)
        return torch.cat([speed, photo], dim=1)

class EnhancedForestEnvironment:
    def __init__(self, num_trees=6, area_size=30.0):
        self.num_trees = num_trees
        self.area_size = area_size
        self.generate_new_forest()
    
    def generate_new_forest(self):
        self.trees = []
        
        for i in range(self.num_trees):
            while True:
                x = random.uniform(5.0, self.area_size)
                y = random.uniform(-self.area_size/2, self.area_size/2)
                
                too_close = False
                for tree in self.trees:
                    dist = np.sqrt((x - tree[0])**2 + (y - tree[1])**2)
                    if dist < 4.0:
                        too_close = True
                        break
                
                if not too_close:
                    self.trees.append([x, y])
                    break
        
        self.trees = np.array(self.trees)
        self.reset()
    
    def reset(self):
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.current_tree_idx = 0
        self.captured_trees = []
        self.total_distance = 0.0
        self.step_count = 0
        self.completed = False
        self.last_photo_step = -100
        self.photo_distances = []
        self.photo_statuses = []
        self.distance_history = []
        self.path_history = [self.position.copy()]
        
        return self._get_observation()
    
    def reset_with_new_forest(self):
        self.generate_new_forest()
        return self.reset()
    
    def _get_observation(self):
        if self.current_tree_idx < len(self.trees):
            tree = self.trees[self.current_tree_idx]
            dx = tree[0] - self.position[0]
            dy = tree[1] - self.position[1]
            distance = np.sqrt(dx**2 + dy**2)
            speed_mag = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            return np.array([dx, dy, distance, speed_mag])
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    def step(self, action):
        speed_command, photo_command = action
        speed_command = float(np.clip(speed_command, 0.0, 1.0))
        photo_command = 1 if photo_command > 0.85 else 0
        
        dt = 0.5
        reward = 0.0
        
        if self.current_tree_idx < len(self.trees):
            tree = self.trees[self.current_tree_idx]
            direction = tree - self.position
            distance = np.linalg.norm(direction)
            self.distance_history.append(distance)
            
            if distance > 0.1:
                direction = direction / distance
            
            max_speed = 1.5
            acceleration = direction * (speed_command * max_speed)
            self.velocity = self.velocity * 0.7 + acceleration * dt
            self.position += self.velocity * dt
            self.path_history.append(self.position.copy())
            self.total_distance += np.linalg.norm(self.velocity * dt)
            
            new_distance = np.linalg.norm(tree - self.position)
            
            distance_reward = (distance - new_distance) * 0.8
            reward += distance_reward
            
            speed_penalty = -0.02
            reward += speed_penalty
            
            if distance > 8.0:
                optimal_speed = 0.95
                speed_error = abs(speed_command - optimal_speed)
                reward -= speed_error * 0.15
            
            elif 4.0 <= distance <= 8.0:
                optimal_speed = 0.4 + 0.4 * (distance - 4.0) / 4.0
                speed_error = abs(speed_command - optimal_speed)
                reward -= speed_error * 0.25
                
                if speed_command < 0.7:
                    reward += 0.15
            
            elif 2.0 <= distance < 4.0:
                optimal_speed = 0.2 + 0.2 * (distance - 2.0) / 2.0
                speed_error = abs(speed_command - optimal_speed)
                reward -= speed_error * 0.4
                
                if speed_command < 0.4:
                    reward += 0.25
            
            elif distance < 2.0:
                optimal_speed = 0.1 * (distance / 2.0)
                speed_error = abs(speed_command - optimal_speed)
                reward -= speed_error * 0.6
                
                if speed_command < 0.2:
                    reward += 0.35
            
            if photo_command == 1:
                if self.step_count - self.last_photo_step < 10:
                    reward -= 15.0
                    self.photo_distances.append(new_distance)
                    self.photo_statuses.append("TOO_SOON")
                
                elif new_distance < 1.2:
                    if self.current_tree_idx not in self.captured_trees:
                        capture_bonus = 30.0 * (1.2 - new_distance)
                        efficiency_bonus = 20.0 * (1.0 / (self.step_count * 0.05 + 1))
                        reward += capture_bonus + efficiency_bonus
                        
                        self.captured_trees.append(self.current_tree_idx)
                        self.photo_distances.append(new_distance)
                        self.photo_statuses.append("GOOD")
                        
                        uncaptured_indices = [i for i in range(len(self.trees)) 
                                            if i not in self.captured_trees]
                        
                        if uncaptured_indices:
                            distances = []
                            for idx in uncaptured_indices:
                                dist = np.linalg.norm(self.trees[idx] - self.position)
                                distances.append((dist, idx))
                            
                            distances.sort()
                            self.current_tree_idx = distances[0][1]
                        else:
                            self.completed = True
                            completion_bonus = 60.0
                            reward += completion_bonus
                        
                        self.last_photo_step = self.step_count
                    else:
                        reward -= 12.0
                        self.photo_distances.append(new_distance)
                        self.photo_statuses.append("DUPLICATE")
                else:
                    distance_penalty = 10.0 + (new_distance - 1.2) * 4.0
                    reward -= distance_penalty
                    self.photo_distances.append(new_distance)
                    self.photo_statuses.append("FAR")
        
        self.step_count += 1
        
        if self.step_count >= 250:
            self.completed = True
        
        return self._get_observation(), reward, self.completed
    
    def get_test_summary(self):
        return {
            'trees_captured': len(self.captured_trees),
            'total_trees': self.num_trees,
            'steps': self.step_count,
            'total_distance': float(self.total_distance),
            'avg_speed': float(self.total_distance / max(self.step_count, 1)),
            'photo_distances': self.photo_distances.copy(),
            'photo_statuses': self.photo_statuses.copy(),
            'min_distance': np.min(self.distance_history) if self.distance_history else 0,
            'avg_distance': np.mean(self.distance_history) if self.distance_history else 0,
            'path_history': self.path_history.copy(),
            'tree_positions': self.trees.copy()
        }

class DetailedEvaluationSystem:
    def __init__(self):
        self.controller = DronePhotographyController()
        self.optimizer = optim.AdamW(self.controller.parameters(), lr=0.001, weight_decay=0.0005)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=50)
        self.loss_fn = nn.MSELoss()
        
    def generate_training_samples(self, num_samples=1500):
        samples = []
        
        for _ in range(num_samples // 4):
            angle = np.random.uniform(0, 2*np.pi)
            
            far_dist = np.random.uniform(10.0, 35.0)
            dx = far_dist * np.cos(angle)
            dy = far_dist * np.sin(angle)
            speed = np.random.uniform(0.0, 1.0)
            distance = np.sqrt(dx**2 + dy**2)
            samples.append(([dx, dy, distance, speed], [0.95, 0.0]))
            
            mid_dist = np.random.uniform(3.0, 10.0)
            dx = mid_dist * np.cos(angle)
            dy = mid_dist * np.sin(angle)
            speed = np.random.uniform(0.0, 1.0)
            distance = np.sqrt(dx**2 + dy**2)
            target_speed = 0.3 + 0.5 * (mid_dist - 3.0) / 7.0
            samples.append(([dx, dy, distance, speed], [target_speed, 0.0]))
            
            near_dist = np.random.uniform(1.0, 3.0)
            dx = near_dist * np.cos(angle)
            dy = near_dist * np.sin(angle)
            speed = np.random.uniform(0.0, 0.5)
            distance = np.sqrt(dx**2 + dy**2)
            target_speed = 0.1 + 0.15 * (near_dist - 1.0) / 2.0
            samples.append(([dx, dy, distance, speed], [target_speed, 0.0]))
            
            photo_dist = np.random.uniform(0.3, 1.2)
            dx = photo_dist * np.cos(angle)
            dy = photo_dist * np.sin(angle)
            speed = np.random.uniform(0.0, 0.2)
            distance = np.sqrt(dx**2 + dy**2)
            photo_prob = max(0.0, 1.0 - (photo_dist - 0.3) / 0.9)
            target_speed = 0.05 * (1.2 - photo_dist)
            samples.append(([dx, dy, distance, speed], [target_speed, photo_prob]))
        
        np.random.shuffle(samples)
        return samples
    
    def train(self, epochs=4000):
        samples = self.generate_training_samples(1500)
        inputs = torch.FloatTensor([s[0] for s in samples])
        targets = torch.FloatTensor([s[1] for s in samples])
        
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.controller.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_inputs, batch_targets in dataloader:
                self.optimizer.zero_grad()
                predictions = self.controller(batch_inputs)
                loss = self.loss_fn(predictions, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            self.scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.controller.state_dict(), "enhanced_drone_controller.pt")
            else:
                patience_counter += 1
            
            if epoch % 400 == 0 or epoch == epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f} | Best: {best_loss:.6f}")
            
            if patience_counter > 100:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"\nTraining completed. Best loss: {best_loss:.6f}")
        return best_loss
    
    def run_comprehensive_evaluation(self, num_tests=25):
        self.controller.load_state_dict(torch.load("enhanced_drone_controller.pt"))
        self.controller.eval()
        
        all_results = []
        test_details = []
        
        for test_num in range(num_tests):
            print(f"\n" + "="*70)
            print(f"TEST {test_num + 1}/{num_tests}")
            print("="*70)
            
            env = EnhancedForestEnvironment(num_trees=6, area_size=30.0)
            observation = env.reset_with_new_forest()
            total_reward = 0.0
            
            tree_positions = env.trees.copy()
            print(f"\nForest configuration:")
            for i, pos in enumerate(tree_positions):
                initial_dist = np.linalg.norm(pos)
                print(f"  Tree {i+1}: Position ({pos[0]:.1f}, {pos[1]:.1f}) | Initial distance: {initial_dist:.1f}m")
            
            step_counter = 0
            detailed_log = []
            
            while not env.completed:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                    action_probs = self.controller(obs_tensor).squeeze().numpy()
                
                speed_action = float(action_probs[0])
                photo_prob = float(action_probs[1])
                photo_action = 1 if photo_prob > 0.85 else 0
                
                observation, reward, completed = env.step([speed_action, photo_action])
                total_reward += reward
                step_counter += 1
                
                current_tree = env.trees[env.current_tree_idx] if env.current_tree_idx < len(env.trees) else None
                if current_tree is not None:
                    current_distance = np.linalg.norm(current_tree - env.position)
                    
                    if photo_action == 1 or step_counter % 25 == 0 or step_counter <= 5:
                        detailed_log.append({
                            'step': step_counter,
                            'position': env.position.copy(),
                            'target_tree': env.current_tree_idx + 1,
                            'distance': float(current_distance),
                            'speed': float(speed_action),
                            'photo': photo_action,
                            'photo_prob': float(photo_prob),
                            'reward': float(reward)
                        })
            
            summary = env.get_test_summary()
            
            test_result = {
                'test_id': test_num + 1,
                'total_reward': total_reward,
                'summary': summary,
                'detailed_log': detailed_log[-8:],
                'tree_positions': tree_positions
            }
            
            all_results.append(test_result)
            test_details.append(test_result)
            
            print(f"\nTest {test_num + 1} Results:")
            print(f"  Trees captured: {summary['trees_captured']}/{summary['total_trees']}")
            print(f"  Total reward: {total_reward:.1f}")
            print(f"  Steps used: {summary['steps']}")
            print(f"  Distance traveled: {summary['total_distance']:.1f}m")
            print(f"  Minimum distance achieved: {summary['min_distance']:.2f}m")
            print(f"  Average distance to targets: {summary['avg_distance']:.2f}m")
            
            if summary['photo_distances']:
                print(f"\n  Photo analysis:")
                for i, (dist, status) in enumerate(zip(summary['photo_distances'], summary['photo_statuses'])):
                    print(f"    Photo {i+1}: {dist:.2f}m - {status}")
                
                good_photos = [dist for dist, status in zip(summary['photo_distances'], summary['photo_statuses']) 
                             if status == "GOOD"]
                far_photos = [dist for dist, status in zip(summary['photo_distances'], summary['photo_statuses']) 
                            if status == "FAR"]
                
                if good_photos:
                    print(f"    Good photos: {len(good_photos)} | Average distance: {np.mean(good_photos):.2f}m")
                if far_photos:
                    print(f"    Far photos: {len(far_photos)} | Average distance: {np.mean(far_photos):.2f}m")
                
                success_rate = len(good_photos) / len(summary['photo_distances'])
                print(f"    Photo success rate: {success_rate:.1%}")
            
            print(f"\n  Last actions:")
            for log in detailed_log[-4:]:
                action_type = f"PHOTO at {log['distance']:.1f}m" if log['photo'] else "no photo"
                print(f"    Step {log['step']}: Dist={log['distance']:.1f}m, Speed={log['speed']:.2f}, {action_type}")
        
        return all_results, test_details

def analyze_performance_statistics(all_results):
    print("\n" + "="*70)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*70)
    
    total_tests = len(all_results)
    success_rates = []
    total_rewards = []
    all_photo_distances = []
    all_good_photo_distances = []
    all_far_photo_distances = []
    steps_per_test = []
    min_distances = []
    
    problematic_tests = []
    
    for test in all_results:
        summary = test['summary']
        
        success_rate = summary['trees_captured'] / summary['total_trees']
        success_rates.append(success_rate)
        total_rewards.append(test['total_reward'])
        steps_per_test.append(summary['steps'])
        min_distances.append(summary['min_distance'])
        
        if summary['photo_distances']:
            all_photo_distances.extend(summary['photo_distances'])
            
            for dist, status in zip(summary['photo_distances'], summary['photo_statuses']):
                if status == "GOOD":
                    all_good_photo_distances.append(dist)
                elif status == "FAR":
                    all_far_photo_distances.append(dist)
        
        if summary['trees_captured'] < summary['total_trees']:
            problematic_tests.append(test['test_id'])
    
    print(f"\nStatistics based on {total_tests} tests:")
    print("-" * 50)
    print(f"  Average success rate: {np.mean(success_rates):.1%}")
    print(f"  Standard deviation: {np.std(success_rates):.3f}")
    print(f"  Average total reward: {np.mean(total_rewards):.1f}")
    print(f"  Average steps per test: {np.mean(steps_per_test):.0f}")
    print(f"  Average minimum distance: {np.mean(min_distances):.2f}m")
    
    if all_photo_distances:
        print(f"\n  Photo statistics:")
        print(f"    Total photos taken: {len(all_photo_distances)}")
        print(f"    Average photo distance: {np.mean(all_photo_distances):.2f}m")
        print(f"    Minimum photo distance: {np.min(all_photo_distances):.2f}m")
        print(f"    Maximum photo distance: {np.max(all_photo_distances):.2f}m")
        
        if all_good_photo_distances:
            print(f"    Good photos: {len(all_good_photo_distances)}")
            print(f"    Average good photo distance: {np.mean(all_good_photo_distances):.2f}m")
        
        if all_far_photo_distances:
            print(f"    Far photos: {len(all_far_photo_distances)}")
            print(f"    Average far photo distance: {np.mean(all_far_photo_distances):.2f}m")
            print(f"    Percentage of far photos: {len(all_far_photo_distances)/len(all_photo_distances):.1%}")
        
        photos_beyond_2m = len([d for d in all_photo_distances if d > 2.0])
        if photos_beyond_2m > 0:
            print(f"\n  PROBLEM DETECTED: Photos beyond 2.0m")
            print(f"    Photos > 2.0m: {photos_beyond_2m}/{len(all_photo_distances)} ({photos_beyond_2m/len(all_photo_distances):.1%})")
            far_dists = [d for d in all_photo_distances if d > 2.0]
            print(f"    Average distance of far photos: {np.mean(far_dists):.1f}m")
    
    if problematic_tests:
        print(f"\n  Tests with missed trees: {problematic_tests}")
        print(f"    {len(problematic_tests)}/{total_tests} tests had incomplete capture")
    
    print(f"\n  Performance categories:")
    excellent_tests = len([r for r in success_rates if r == 1.0])
    good_tests = len([r for r in success_rates if 0.8 <= r < 1.0])
    poor_tests = len([r for r in success_rates if r < 0.8])
    
    print(f"    Excellent (100%): {excellent_tests}/{total_tests}")
    print(f"    Good (80-99%): {good_tests}/{total_tests}")
    print(f"    Poor (<80%): {poor_tests}/{total_tests}")
    
    if total_tests >= 10:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(range(1, total_tests + 1), success_rates, 'bo-')
        axes[0, 0].set_title('Success Rate per Test')
        axes[0, 0].set_xlabel('Test Number')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=np.mean(success_rates), color='r', linestyle='--', 
                          label=f'Average: {np.mean(success_rates):.1%}')
        axes[0, 0].legend()
        
        axes[0, 1].hist(all_photo_distances if all_photo_distances else [0], bins=20, edgecolor='black')
        axes[0, 1].set_title('Photo Distance Distribution')
        axes[0, 1].set_xlabel('Distance (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=1.2, color='r', linestyle='--', label='Target: 1.2m')
        axes[0, 1].legend()
        
        axes[1, 0].plot(range(1, total_tests + 1), total_rewards, 'go-')
        axes[1, 0].set_title('Total Reward per Test')
        axes[1, 0].set_xlabel('Test Number')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(range(1, total_tests + 1), min_distances, 'ro-')
        axes[1, 1].set_title('Minimum Distance Achieved per Test')
        axes[1, 1].set_xlabel('Test Number')
        axes[1, 1].set_ylabel('Distance (m)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=1.2, color='g', linestyle='--', label='Target: 1.2m')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('drone_performance_analysis.png', dpi=300)
        print(f"\nAnalysis graphs saved to: drone_performance_analysis.png")
    
    return {
        'success_rates': success_rates,
        'total_rewards': total_rewards,
        'photo_distances': all_photo_distances,
        'good_photo_distances': all_good_photo_distances,
        'far_photo_distances': all_far_photo_distances,
        'problematic_tests': problematic_tests,
        'avg_success_rate': np.mean(success_rates),
        'avg_photo_distance': np.mean(all_photo_distances) if all_photo_distances else 0
    }

def main():
    print("=" * 70)
    print("ENHANCED DRONE PHOTOGRAPHY EVALUATION SYSTEM")
    print("=" * 70)
    
    system = DetailedEvaluationSystem()
    
    print("\n[PHASE 1] Extended Training")
    print("-" * 50)
    final_loss = system.train(epochs=4000)
    
    print("\n[PHASE 2] Comprehensive Evaluation (25 tests)")
    print("-" * 50)
    
    all_results, test_details = system.run_comprehensive_evaluation(num_tests=25)
    
    stats = analyze_performance_statistics(all_results)
    
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)
    
    if stats['problematic_tests']:
        print(f"\nISSUES IDENTIFIED:")
        print(f"  Tests with missed trees: {len(stats['problematic_tests'])}/25")
        print(f"  Overall success rate: {stats['avg_success_rate']:.1%}")
        
        if stats['photo_distances']:
            avg_photo_dist = stats['avg_photo_distance']
            far_photos_count = len(stats['far_photo_distances'])
            total_photos = len(stats['photo_distances'])
            
            print(f"\n  PHOTO DISTANCE ANALYSIS:")
            print(f"    Average photo distance: {avg_photo_dist:.2f}m")
            print(f"    Target photo distance: <1.2m")
            
            if avg_photo_dist > 1.5:
                print(f"    MAJOR ISSUE: Photos taken too far on average")
                print(f"    Far photos (>1.2m): {far_photos_count}/{total_photos} ({far_photos_count/total_photos:.1%})")
            elif avg_photo_dist > 1.2:
                print(f"    MODERATE ISSUE: Photos slightly too far")
                print(f"    Improvement needed in photo distance")
            else:
                print(f"    GOOD: Photo distance within target range")
    else:
        print(f"\nALL TESTS COMPLETED SUCCESSFULLY")
        print(f"  Success rate: 100%")
        if stats['photo_distances']:
            avg_photo_dist = stats['avg_photo_distance']
            print(f"  Average photo distance: {avg_photo_dist:.2f}m")
    
    print(f"\nModel saved: enhanced_drone_controller.pt")
    print(f"Analysis saved: drone_performance_analysis.png")
    
    return stats

if __name__ == "__main__":
    performance_stats = main()