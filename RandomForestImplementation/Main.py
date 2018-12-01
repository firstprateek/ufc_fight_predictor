import csv
import os
from Data import *

def main():
    # generate path to data file
    dir_path = os.getcwd().replace('RandomForestImplementation', 'data.csv')

    data = Data()
    # using csv library to read the data file
    with open(dir_path) as file:
        temp_data = []
        fight_data = csv.reader(file)
        
        # Set None values to 0
        for row in fight_data:
            for col_idx, col in enumerate(row):
                if col_idx not in string_columns and not row[col_idx]:
                    row[col_idx] = 0 #def_num
            temp_data.append(row)

        data.set_headers(temp_data[0])
        data.standardize_headers() # standardize the header names

        temp_data.pop(0)
        
        # remove rows with no contest
        for i in range(len(temp_data)):
            if temp_data[i][-1] is "no contest":
                temp_data.pop(i)

        # build non header rows for data
        # congregate all other rounds into round 1
        for r in range(len(temp_data)):
            row = []
            for c in range(len(temp_data[0])):
                if 9 <= c <= 95 or 458 <= c <= 544:
                    index = 1
                    total = 0
                    while index <= 5:
                        total = total + int(float(temp_data[r][c + (index - 1) * 87]))
                        index = index + 1
                    row.append(total)
                else:
                    row.append(temp_data[r][c])
            data.add_row(row)

        # calculate averages for winners and losers
        loser_averages, winner_averages = {}, {}
        data_rows = data.rows
        data_headers = data.headers

        for row_idx, row_val in enumerate(data_rows):
            # These are the rows which will be set to average values
            # As all their values were None
            if sum(row_val[9:96]) + sum(row_val[458:545]) == 0:  continue

            for col_idx, col_val in enumerate(row_val):
                if not (9 <= col_idx <= 95 or 458 <= col_idx <= 544): continue

                header_name = data_headers[col_idx]

                if (header_name[0] == 'B' and row_val[-1] == "blue") or \
                    (header_name[0] == 'R' and row_val[-1] == "red"):

                    if header_name not in winner_averages:
                        winner_averages[header_name] = {}
                        winner_averages[header_name]['total'] = col_val
                        winner_averages[header_name]['count'] = 1
                    else:
                        winner_averages[header_name]['total'] += col_val
                        winner_averages[header_name]['count'] += 1
                else:
                    if header_name not in loser_averages:
                        loser_averages[header_name] = {}
                        loser_averages[header_name]['total'] = col_val
                        loser_averages[header_name]['count'] = 1
                    else:
                        loser_averages[header_name]['total'] += col_val
                        loser_averages[header_name]['count'] += 1

        # print(f'winner averages: {winner_averages}')
        # print(f'loser averages: {loser_averages}')

        for key, value in winner_averages.items():
            winner_averages[key] = value['total'] // value['count']
        
        for key, value in loser_averages.items():
            loser_averages[key] = value['total'] // value['count']

        # Replace by average:
        for row_idx, row_val in enumerate(data_rows):
            if sum(row_val[9:96]) + sum(row_val[458:545]) != 0: continue
            
            for col_idx, col_val in enumerate(row_val):
                if not (9 <= col_idx <= 95 or 458 <= col_idx <= 544): continue
                
                header_name = data_headers[col_idx]
                last_round = int(float(row_val[447]))
                max_round = int(float(row_val[448]))

                if (header_name[0] == 'B' and row_val[-1] == "blue") or \
                    (header_name[0] == 'R' and row_val[-1] == "red"):
                    row_val[col_idx] = (winner_averages[header_name] * last_round) // max_round
                else:
                    row_val[col_idx] = (loser_averages[header_name] * last_round) // max_round

        # maybe remove more unneccessary columns
        data.purge_columns()

    new_data = open('data_out3.csv', 'w')
    writer = csv.writer(new_data, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL)

    writer.writerow(data.headers)
    for i in data.rows:
        writer.writerow(i)

if __name__ == '__main__':
    main()
