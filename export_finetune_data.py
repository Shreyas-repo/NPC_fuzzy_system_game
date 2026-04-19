#!/usr/bin/env python3
"""Export fine-tuning datasets from conversation_logs.csv."""
import json

from ai.conversation_learning import ConversationLearningModel


def main():
    model = ConversationLearningModel()
    result = model.export_fine_tune_datasets(force=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
