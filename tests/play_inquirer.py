from pprint import pprint
import inquirer

# '''单选'''
# questions = [
#     inquirer.List(
#         "size",
#         message="What size do you need?",
#         choices=["Jumbo", "Large", "Standard", "Medium", "Small", "Micro"],
#     ),
# ]

# answers = inquirer.prompt(questions)
# pprint(answers)

'''多选'''
questions = [
    inquirer.Checkbox(
        "sizes",
        message="Select sizes you need (use space to select, enter to confirm):",
        choices=["Jumbo", "Large", "Standard", "Medium", "Small", "Micro"],
    ),
]

answers = inquirer.prompt(questions)
pprint(answers)
