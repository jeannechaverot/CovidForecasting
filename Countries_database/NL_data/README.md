# nl-corona
Quick-and-dirty scraper to load csv data from the [RIVM Corona map](https://www.rivm.nl/coronavirus-kaart-van-nederland) of the Netherlands. 

Currently just stores the data in a csv file named after the data's timestamp. 

The script will be expanded in the forseeable future to merge and process data from different timestamps, and analyze it. 
After these additions, unit tests will be added.

# Automated execution

The script can be automatically executed numerous ways. The easiest is using a `.bat` file and Windows' built-in Task Scheduler.
Create a `.bat` file with the following contents:

```
cd "C:\...\...\nl-corona\"
python main.py
```

Adjust the path so that it points to your copy of this repository. If using `python3` instead of `python` in your system environment variables (`PATH`), edit the command accordingly. 

In the Task Scheduler, create a new task with `Action: Start a program`. Select the `.bat` file as program to execute. No further parameters are required. All other settings can be configured as you please.
