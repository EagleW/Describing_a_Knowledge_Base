import sys


def get_file_list_from_dir(datadir):
    data = []
    i = 0
    with open(datadir) as f:
        for line in f:
            data.append(line)
            sys.stdout.write(
                '%d line processed\r' % (i)
            )
            i += 1

    return data


def split(file_list):
    num = len(file_list)
    train = num // 10 * 8
    valid = train + (num - train) // 2
    training = file_list[:train]
    validing = file_list[train:valid]
    testing = file_list[valid:]
    return training, validing, testing


def write_files(filename, data):
    f = open(filename, "w")
    f.writelines(data)
    f.flush()
    f.close()


if __name__ == "__main__":
    path = '../Wiki_dataset/'
    train, valid, test = split(get_file_list_from_dir(path + 'wiki_animal.json'))
    filename = path + 'train_wiki_A.json'
    write_files(filename, train)
    print('Finished Animal train')
    filename = path + 'valid_wiki_A.json'
    write_files(filename, valid)
    print('Finished Animal valid')
    filename = path + 'test_wiki_A.json'
    write_files(filename, test)
    print('Finished Animal test')
    train, valid, test = split(get_file_list_from_dir(path + 'wiki_person.json'))
    filename = path + 'train_wiki_P.json'
    write_files(filename, train)
    print('Finished Person train')
    filename = path + 'valid_wiki_P.json'
    write_files(filename, valid)
    print('Finished Person valid')
    filename = path + 'test_wiki_P.json'
    write_files(filename, test)
    print('Finished Person test')