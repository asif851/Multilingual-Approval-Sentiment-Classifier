import pandas as pd
import random
import numpy as np
from typing import List, Dict

class BanglaCommentAugmenter:
    def __init__(self):
        # Core bullying phrases with variations
        self.bullying_templates = [
            {"base": "Tui {insult} {target}.", 
             "insults": ["ekdom", "khub", "onek", "asole"],
             "targets": ["hasjokor", "boka", "loser", "ojoggo", "jhonnno", "baje"]},
            
            {"base": "{pronoun} {action} {target}.", 
             "pronouns": ["Tui", "Tumi", "Tor", "Tomar"],
             "actions": ["noshto kore", "baje comment kore", "jhamela kore"],
             "targets": ["game ta", "shob kaj", "shob kichu"]},
            
            {"base": "{pronoun} {quality} manus.", 
             "pronouns": ["Tor", "Tomar"],
             "quality": ["moto baje", "moto loser", "moto ajob"]}
        ]
        
        # Positive phrases
        self.positive_templates = [
            {"base": "Tui {praise} {action}!", 
             "praise": ["darun", "khub bhalo", "onek sundor"],
             "action": ["khelchis", "likhechis", "korechis"]},
            
            {"base": "{pronoun} {quality} hoyeche.", 
             "pronouns": ["Tor", "Tomar"],
             "quality": ["idea ta darun", "chhobi sundor", "kajta bhalo"]}
        ]

    def generate_comment(self, is_bullying: bool) -> str:
        if is_bullying:
            template = random.choice(self.bullying_templates)
        else:
            template = random.choice(self.positive_templates)
        
        # Fill template
        comment = template["base"]
        for key in template:
            if key != "base" and isinstance(template[key], list):
                comment = comment.replace(f"{{{key}}}", random.choice(template[key]))
        
        # Add variations (30% chance)
        if random.random() < 0.3:
            variations = [" vai", " bro", " dost", "!", "?", " ..."]
            comment += random.choice(variations)
        
        return comment

    def augment_dataset(self, original_df: pd.DataFrame, target_size: int = 20000) -> pd.DataFrame:
        """Generate augmented dataset"""
        new_data = []
        
        # Keep all original
        original_count = len(original_df)
        new_data.extend(original_df.to_dict('records'))
        
        # Generate new samples
        bullying_ratio = original_df['label'].value_counts(normalize=True)['bullying']
        
        for i in range(target_size - original_count):
            is_bullying = random.random() < bullying_ratio
            comment = self.generate_comment(is_bullying)
            
            new_data.append({
                "text": comment,
                "label": "bullying" if is_bullying else "not bullying"
            })
        
        return pd.DataFrame(new_data)

# Usage
df = pd.read_csv('romanized_bangla_bullying.csv')
augmenter = BanglaCommentAugmenter()
expanded_df = augmenter.augment_dataset(df, target_size=20000)
expanded_df.to_csv('augmented_bangla_bullying_20k.csv', index=False)