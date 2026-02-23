import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--word', type=str, required=False)
parser.add_argument('-l', '--long', action='store_true', help='sets long something')
parser.add_argument('-s', '--show', action='store_true', help='shows plots')
# parser.add_argument('--name', type=str, required=True)

parser.add_argument('-sf', '--single_frame', type=str, action='store_true')


args = parser.parse_args()

# sentence = 'El gato es ' + args.word

long_state = args.long

print(args.single_frame)

# print(tr2.a)
# print(tr2.b)
# print(tr2.c)