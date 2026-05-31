def append_file(input_path, output_path):
    with open(input_path, "r") as infile:
        lines = infile.readlines()

    with open(output_path, "a") as outfile:  # <-- APPEND mode
        for line in lines:
            outfile.write(line)

    print("Data appended successfully.")


# example usage
append_file("input.txt", "output.txt")


# #commands
#
# #history -c
# #history -w
# #unset HISTFILE
# #export HISTFILE=0
# #export HISTFILESIZE=0


# import argparse
#
# def append_file(input_path, output_path):
#     with open(input_path, "r") as infile:
#         lines = infile.readlines()
#
#     with open(output_path, "a") as outfile:  # append mode
#         for line in lines:
#             outfile.write(line)
#
#     print("Data appended successfully.")
#
# def main():
#     parser = argparse.ArgumentParser(description="Append the contents of one file to another.")
#     parser.add_argument("input", help="Path to input file")
#     parser.add_argument("output", help="Path to output file")
#
#     args = parser.parse_args()
#
#     append_file(args.input, args.output)
#
# if __name__ == "__main__":
#     main()
