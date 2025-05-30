import csv
import os
import json

class ImageEvalLogger:
    def __init__(self, filepath="image_eval_log.csv"):
        self.filepath = filepath
        self.header = ["target_image_url", "overall_score", "original_description", "suggested_description"]
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def log(self, target_image_url, overall_score, original_description, suggested_description):
        with open(self.filepath, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                target_image_url,
                overall_score,
                original_description,
                suggested_description
            ])
        print(f"Logged: {target_image_url}")
        
class HistoryLogger:
    '''
    Enum: (Style, Context, Final Summary)
    '''
    def __init__(self, filepath="debate_history_log.json"):
        self.filepath = filepath
        self.logger_dict = {
            "Style": "",
            "Context": "",
            "Final Summary": "",
        }
        self.data = []
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def log_history(self, target_image_url, total_rounds):
        entry = {
            "target_image_url": target_image_url,
            "total_rounds": total_rounds,
            "critique_scorer_history": self.logger_dict["Style"],
            "analyzer_scorer_history": self.logger_dict["Context"],
            "final_summary": self.logger_dict["Final Summary"]
        }
        self.data.append(entry)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"HistoryLogger saved: {target_image_url}")