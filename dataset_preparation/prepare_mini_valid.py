import os

def main():
    create_mini_valid()

def create_mini_valid():
    data_dir = '/data2/longshen/musecoco_data/datasets'
    valid_fp = os.path.join(data_dir, 'valid.txt')
    mini_valid_fp = os.path.join(data_dir, 'mini_valid.txt')
    
    # Read valid.txt
    with open(valid_fp, 'r') as f:
        lines = f.readlines()
    
    # Random select 200 lines
    import random
    random.seed(0)
    random.shuffle(lines)
    mini_lines = lines[:200]

    # Write mini_valid.txt
    with open(mini_valid_fp, 'w') as f:
        for line in mini_lines:
            f.write(line)


if __name__ == '__main__':
    main()