import logging
import os
import sys
import lda_analysis

directory_in_str =os.getcwd()
directory_in_str += "\clean"
directory = os.fsencode(directory_in_str)

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        for x in range(1,6):
            print("Analysing file: " + filename[6:-4] + " Rating: %s" %x)
            try:
                lda_analysis.main(filename[6:-4], x)
                print("Analysis complete! Moving to next one. \n")
            except Exception:
                logging.info("%s_%s Did not work. Moving on to next one.."%(filename,x))
        try:
            lda_analysis.main(filename[6:-4], "all")
        except Exception:
            logging.info("%s_all Did not work. Moving on to next one..\n"%(filename))
    else:
        continue
