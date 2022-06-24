"""
replace "template" in config
"""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert template to specific')
    parser.add_argument('--org-config', default='', help='origin config file (.py)')
    parser.add_argument('--new-config', default='', help='new config file (.py)')
    parser.add_argument('--seed', default='0', type=str)
    parser.add_argument('--percent', default='1', type=str)
    parser.add_argument('--gpu', default='2', type=str)
    parser.add_argument('--times', default='1', type=str)
    parser.add_argument('--score', default='0.7', type=str)
    parser.add_argument('--data', default='coco-standard', type=str)

    args = parser.parse_args()

    with open(args.org_config, 'r', encoding='utf-8') as fr, open(args.new_config, 'w', encoding='utf-8') as fw:
        for line in fr:
            if 'seed_template' in line:
                line = line.replace('seed_template', args.seed)
            if 'percent_template' in line:
                line = line.replace('percent_template', args.percent)
            if 'gpu_template' in line:
                line = line.replace('gpu_template', args.gpu)
            if 'times_template' in line:
                line = line.replace('times_template', args.times)
            if 'score_template' in line:
                line = line.replace('score_template', args.score)
            if 'data_template' in line:
                value = f'\'{args.data}\''
                line = line.replace('data_template', value)
            fw.write(line)
