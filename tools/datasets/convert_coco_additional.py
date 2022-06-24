"""
add empty information to coco additional
"""
import argparse
import mmcv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add empty information to coco additional')
    parser.add_argument('--additional-json',
                        default='',
                        help='coco additional json')
    parser.add_argument('--standard-json', default='',
                        help='one of the labeled data json file')
    parser.add_argument('--output-json', default='',
                        help='output json file')

    args = parser.parse_args()

    additional_info = mmcv.load(args.additional_json)
    standard_info = mmcv.load(args.standard_json)

    additional_info['annotations'] = []
    additional_info['categories'] = standard_info['categories']

    mmcv.dump(additional_info, args.output_json)
