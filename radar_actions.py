

from datetime import datetime
import csv
def on_target_acquired(recent_speed):
    now = datetime.now()
    print(f'on_target_acquired called at velocity:{recent_speed} mph {now.strftime("%H:%M:%S")}')

 #Write CSV file
    with open('radar_file.csv', mode='a') as radar_file:
        radar_writer = csv.writer(radar_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        radar_writer.writerow([f'{now.strftime("%-H:%-M:%-S")}',f'{abs(recent_speed)}'])

def on_target_lost():
    now = datetime.now()
    print(f'on_target_lost called at {now.strftime("%H:%M:%S")}')

def on_idle_notice_interval():
    now = datetime.now()
    print(f'on_target_lost called at {now.strftime("%H:%M:%S")}')
