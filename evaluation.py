import os

main_dir = os.getcwd()
result_dir = os.path.join(main_dir, "test_result")

f = [] # Store filenames - character
for (dirpath, dirnames, filenames) in os.walk(result_dir):
    f.extend(filenames)
    break

with open("result_statistic.txt", "w") as result_file:
    for filename in f:
        line_count = 0
        correct_count = 0
        text_file = os.path.join(result_dir, filename)
        with open(text_file, 'r') as file:
            for line in file:
                # print(line)
                line_count += 1
                if filename[0] == line[0]:
                    correct_count += 1
        accuracy = correct_count / line_count
        result_file.write("Letter :" + filename)
        result_file.write(
            " Total sample: {0} - Correct prediction {1} - Accuracy: {2}\n".format(line_count,
                                                                                   correct_count,
                                                                                   str(accuracy)
                                                                                   )
        )
