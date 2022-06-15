import subprocess

program_list = ['prepare.py', 'featurization.py', 'train.py']

for program in program_list:
    if program == "prepare.py":
        subprocess.call(['python', program, 'train.csv', 'test.csv'])
    else:
        subprocess.call(['python', program])
    print("Finished:" + program)