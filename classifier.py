from tqdm import tqdm
import json

# try:
#     with open('settings.json') as f:
#         settings = json.load(f)                         # получаем данные файла настроек
# except IOError:
#     print('File "settings.json" is MISSING.')
#     wait = input("PRESS ENTER TO EXIT.")
#     raise SystemExit(1)
#
#
# WORK_DIRECTORY = settings['work_dir']
# POSITIVE_SET = settings['positive_set']
# NEGATIVE_SET = settings['negative_set']

WORK_DIRECTORY = "D:\Downloads\ComputerVision_test"
POSITIVE_SET = "00_hardworkers"
NEGATIVE_SET = "01_lazyobnes"

def main():
    print('Start classifier')
    # print('Work Dir: {}\n'
    #       'Positive Set: {}\n'
    #       'Negative Set: {}'.format(WORK_DIRECTORY, POSITIVE_SET, NEGATIVE_SET))


if __name__ == '__main__':
    main()

