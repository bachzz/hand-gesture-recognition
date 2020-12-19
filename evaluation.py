import os
import collections
main_dir = os.getcwd()
result_dir = os.path.join(main_dir, "test_result")

f = [] # Store filenames - character
for (dirpath, dirnames, filenames) in os.walk(result_dir):
    f.extend(filenames)
    break
result = {}
with open("result_statistic.txt", "w") as result_file:
    for filename in f:
        # print(filename)
        line_count = 0
        correct_count = 0
        text_file = os.path.join(result_dir, filename)
        with open(text_file, 'r') as file:
            for line in file:
                # print(line)
                line_count += 1
                if filename[0] == line[0]:
                    correct_count += 1
        if line_count == 0:
            continue
        accuracy = correct_count / line_count
        result_file.write("Letter :" + filename)
        single_result = { filename : accuracy}
        result.update(single_result)
        result_file.write(
            " Total sample: {0} - Correct prediction {1} - Accuracy: {2}\n".format(line_count,
                                                                                   correct_count,
                                                                                   str(accuracy)
                                                                                   )
        )

ordered_result = collections.OrderedDict(sorted(result.items()))
count=0;
for k,v in ordered_result.items():
    if v < 0.8:
        count += 1
        print("Character : {0} - Accuracy {1}".format(k[0],v))
print(" Total : {}".format(str(count)))

sort_value = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}

for k,v in sort_value.items():
    print("Character : {0} - Accuracy {1}".format(k[0],v))
