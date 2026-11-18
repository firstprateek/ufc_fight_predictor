# number of columns in the dataset of a fighter
number_att_of_a_fighter = 444
number_of_att = 96
# default number for None values
def_num = 999

string_columns = {
    4,
    6, #
    7,
    444,
    453,
    455, #
    456,
    893,
    894,
}

'''
458 is where red R_ starts <= 544
893, 894 is last 2 (if not deleted)
447 is last round useful in division
by max round which is 448
'''

# Index of unneccessary columns
columns_to_be_deleted = {
    4, # B_HomeTown
    7, # B_Name
    444, # Date
    445, # Event_ID
    446, # Fight_ID
    447, # Last_round
    453, # R_HomeTown
    456, # R_Name
}

for num in range(96, 444):
    columns_to_be_deleted.add(num)

for num in range(545, 894):
    columns_to_be_deleted.add(num)

# Also add other ones like other than total strikes
# ---- Fill this up after generating the other file

# Figure out a way to pluck column_ids using names of files
# whose name isn't on the list of columns to be included

column_suffixes = [
    "_Total_Grappling_Reversals_Landed",
    "_Total_Grappling_Standups_Landed",
    "_Total_Grappling_Submissions_Attempts",
    "_Total_Grappling_Takedowns_Attempts",
    "_Total_Grappling_Takedowns_Landed",
    "_Total_Strikes_Body Significant Strikes_Attempts",
    "_Total_Strikes_Body Significant Strikes_Landed",
    "_Total_Strikes_Body Total Strikes_Attempts",
    "_Total_Strikes_Body Total Strikes_Landed",
    "_Total_Strikes_Clinch Total Strikes_Attempts",
    "_Total_Strikes_Clinch Total Strikes_Landed",
    "_Total_Strikes_Distance Strikes_Attempts",
    "_Total_Strikes_Distance Strikes_Landed",
    "_Total_Strikes_Ground Total Strikes_Attempts",
    "_Total_Strikes_Ground Total Strikes_Landed",
    "_Total_Strikes_Head Total Strikes_Attempts",
    "_Total_Strikes_Head Total Strikes_Landed",
    "_Total_Strikes_Kicks_Attempts",
    "_Total_Strikes_Kicks_Landed",
    "_Total_Strikes_Knock Down_Landed",
    "_Total_Strikes_Legs Total Strikes_Attempts",
    "_Total_Strikes_Legs Total Strikes_Landed",
    "_Total_Strikes_Punches_Attempts",
    "_Total_Strikes_Punches_Landed",
    "_Total_Strikes_Total Strikes_Attempts",
    "_Total_Strikes_Total Strikes_Landed",
    "_Total_TIP_Back Control Time",
    "_Total_TIP_Clinch Time",
    "_Total_TIP_Control Time",
    "_Total_TIP_Distance Time",
    "_Total_TIP_Ground Control Time",
    "_Total_TIP_Ground Time",
    "_Total_TIP_Guard Control Time",
    "_Total_TIP_Half Guard Control Time",
    "_Total_TIP_Misc. Ground Control Time",
    "_Total_TIP_Mount Control Time",
    "_Total_TIP_Neutral Time",
    "_Total_TIP_Side Control Time",
    "_Total_TIP_Standing Time",
    "Prev",
    "Streak",
    "_Age",
    "_Height",
    "_ID",
    "_Weight",
    "_Location",
]

final_columns = [ name for suffix in column_suffixes for name in [f'B{suffix}', f'R{suffix}']]

final_columns.extend([
    "Max_round",
    "winner",
])

class Data:
    def __init__(self):
        self.headers = []
        self.rows = []

    def add_row(self, row):
        self.rows.append(row)

    def set_headers(self, header_list):
        self.headers = header_list

    def standardize_headers(self):
        new_headers = []
        for i in range(len(self.headers)):
            b_prefix = "B_Total_"
            r_prefix = "R_Total_"

            if 8 < i <= 95:
                new_header_name = b_prefix + self.headers[i][10:]
                new_headers.append(new_header_name)
            elif 458 <= i <= 545:
                new_header_name = r_prefix + self.headers[i][10:]
                new_headers.append(new_header_name)
            else:
                new_headers.append(self.headers[i])
        
        self.headers = [header.strip() for header in new_headers]

    def purge_columns(self):
        for del_index in reversed(list(columns_to_be_deleted)):
            self.headers.pop(del_index)
            for row in self.rows: row.pop(del_index)

        temp_rows = []
        col_ids = [ self.headers.index(col_name) for col_name in final_columns]
        
        for row in self.rows:
            temp_rows.append([ row[col_id] for col_id in col_ids ])

        self.headers = final_columns
        self.rows = temp_rows
