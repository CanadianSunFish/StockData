from datetime import date, datetime, timedelta

stock_list = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA', 'SCHL', 'JPM', 'AG', 'DE', 'BCS',
              'EPAM', 'JNJ', 'CVS', 'GM', 'GE', 'MA', 'EXPE', 'MAR', 'BCRX', 'BAM', 'DAL',
              'NKE', 'NCLH', 'KO', 'IBM', 'ABG', 'HEI', 'FSLR', 'MDLZ', 'PSX', 'CAT', 'ICE']
holidays = [date(2022, 4, 15), date(2022, 5, 30), date(2022, 6, 20), date(2022, 7, 3), date(2022, 7, 4),
            date(2022, 9, 5),
            date(2022, 11, 24),
            date(2022, 11, 25), date(2022, 12, 26)]

date_today = date.today()
df_today = str(date_today.strftime('%Y-%m-%d'))
today = date_today.strftime("%Y-%m-%d")
today_ = today
weekday = date_today.isoweekday()

lastdays = [date_today - timedelta(days=day) for day in range(6) if
            (int((date_today - timedelta(days=day)).isoweekday()) < 6) and ((date_today - timedelta(days=day)) not in holidays)]
nextdays = [date_today + timedelta(days=day) for day in range(80) if
            (int((date_today + timedelta(days=day)).isoweekday()) < 6) and ((date_today + timedelta(days=day)) not in holidays)]

tomorrow = nextdays[1]
tomorrow_ = tomorrow.strftime("%d/%m/%Y")
two_days = nextdays[2]
three_days = nextdays[3]
five_days = nextdays[5]
ten_days = nextdays[10]
one_week = nextdays[7]
two_weeks = nextdays[12]
one_month = nextdays[21]
two_months = nextdays[42]

if date_today.isoweekday() < 6 and date_today not in holidays:
    yesterday = lastdays[1]
else:
    yesterday = lastdays[0]
df_yesterday = yesterday.strftime("%Y-%m-%d")
yesterday_ = yesterday.strftime("%Y-%m-%d")
two_ago = lastdays[2]
two_ago_ = two_ago.strftime("%d/%m/%Y")

time_ = datetime.now()
current_time = time_.strftime("%H:%M:%S")
current_minute = int(time_.strftime("%M"))
current_hour = int(time_.strftime("%H"))
ten_ago = ('{}:{}'.format(current_hour, current_minute - 10))


