#!/usr/bin/env python
import csv
import argparse as ap

def main(filename):
    """
    Read in file line by line and output each value trimmed of white space
    for monetDB
    """
    new_filename = filename[:-4] + '_modified.csv'
    rf = open(filename,'r')
    wf = open(new_filename,'w')
    writer = csv.writer(wf)


    for row in rf:
        fields = row.split(',')
        fields = [field.strip() for field in fields]
        writer.writerow(fields)



    rf.close()
    wf.close()
    


if __name__ == '__main__':
    PARSER = ap.ArgumentParser()
    PARSER.add_argument('filename', help='csv file to trim whitespace from')
    ARGS = PARSER.parse_args()
    main(ARGS.filename)
