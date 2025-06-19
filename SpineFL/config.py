import yaml
import random

global cfg
if 'cfg' not in globals():
    with open('config.yml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)


user_distribution = cfg.get('user_distribution', {"rich": 0.3, "normal": 0.4, "low": 0.3})
total_users = cfg.get('numbers', 100)


num_rich = int(user_distribution["rich"] * total_users)
num_normal = int(user_distribution["normal"] * total_users)
num_low = total_users - num_rich - num_normal


user_indices = list(range(total_users))
random.shuffle(user_indices)
cfg["rich_users"] = user_indices[:num_rich]
cfg["normal_users"] = user_indices[num_rich:num_rich + num_normal]
cfg["low_users"] = user_indices[num_rich + num_normal:]


cfg['flexfl'] = {
    'temperature': 3.0,
    'distill_epochs': 3,
    'grad_clip': 1.0,
    'selection_mode': 'adaptive'
}

cfg['alpha']=1
