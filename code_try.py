import argparse






parser = argparse.ArgumentParser()
parser.add_argument('--word', type=str, required=False)
parser.add_argument('-l', '--long', action='store_true', help='sets long something')
parser.add_argument('-s', '--show', action='store_true', help='shows plots')
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()

sentence = 'El gato es ' + args.word

print(args.show)