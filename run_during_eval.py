import os
from time import sleep
import argparse 

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
    
  parser.add_argument(
      '-c',
      "--checkpoint_dir",
      required=True,
      help= "path to folder with checkpoints")

  parser.add_argument(
      '-e',
      "--event_dir",
      required=True,
      help= "path to folder with event")

  args = parser.parse_args()
  
  eventFile = None
  lastUpdateTime = None
  checkpointNumbers = []
  
  filenames = os.listdir( args.checkpoint_dir )
  for filename in filenames:
    if filename.endswith('index'):
      checkpointNumbers.append( filename[11:-6] )
  
  checkpointFile = os.path.join(args.checkpoint_dir, 'checkpoint')
  checkpointNumbers = [int(i) for i in checkpointNumbers]
  checkpointNumbers.sort()
  checkpointNumbers = [str(i) for i in checkpointNumbers]
  print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n', checkpointFile, '\n', checkpointNumbers)
  
  for checkpointNumber in checkpointNumbers:

    f = open(checkpointFile, 'r')
    f_str = f.read()
    f.close()

    f_start = f_str[: f_str.find('model.ckpt-')+11]
    f_end = f_str[f_str.find('\n'):]

    f = open(checkpointFile, 'w')
    f.write( f_start + checkpointNumber + '\"' + f_end)
    f.close()

    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n', checkpointNumber)
    
    if eventFile == None:
      while not os.path.exists(args.event_dir):
        sleep(2)
      while not os.listdir(args.event_dir):
        sleep(2)
      eventFile = os.path.join( args.event_dir, os.listdir(args.event_dir)[0] )
    
    lastUpdateTime = os.stat(eventFile).st_mtime
    print('Checked if event file updated')

    while os.stat(eventFile).st_mtime == lastUpdateTime:
      sleep(5)

