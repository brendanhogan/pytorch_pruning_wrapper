# All runs

# CIFAR100
python main.py -lr 1 -v -ds cifar100
python main.py -lr 1 -v -ds cifar100 -rp
python main.py -lr 1 -v -ds cifar100 -ri -rp

# SVHN
python main.py -lr 1 -v -ds svhn
python main.py -lr 1 -v -ds svhn -rp
python main.py -lr 1 -v -ds svhn -ri -rp
