import sys
import os
import argparse

import pygame
from pygame.locals import *

class game():
    
    def __init__(self, notFlaggedPath, flaggedPath, txtPath, outputFileName):
        
        # Set up key game variables
        self.fileNames = []
        self.currFileInd = 0
        self.txtOutput = []
        self.notFlaggedInd = 0
        self.outputFileName = outputFileName

        # Append all file paths
        if not os.path.exists(flaggedPath):
            print('Path %s does not exist' %flaggedPath)
        if not os.path.exists(notFlaggedPath):
            print('Path %s does not exist' %notFlaggedPath)
        
        currFileNames = os.listdir(notFlaggedPath)
        self.fileNames += [ os.path.join(notFlaggedPath, i) for i in currFileNames ]
        
        self.notFlaggedInd = len(self.fileNames)

        currFileNames = os.listdir(flaggedPath)
        self.fileNames += [ os.path.join(flaggedPath, i) for i in currFileNames ]

        print(self.notFlaggedInd)

        # Read txt
        if os.path.exists(txtPath):
            f = open(txtPath, 'r')
            for filePath in f:
                self.txtOutput.append(filePath)
            f.close()

        # Initialise PyGame.
        pygame.init()

        # Initialise text and image
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.text = self.font.render('Script flagged: %s' %self.currFileInd, True, (0,255,0), (0,0,0))
        self.img = pygame.image.load( self.fileNames[self.currFileInd] )


    def writeOutputCsv(self):
        """ Output csv file according to txtOutput """

        f = open( self.outputFileName, 'w' )
        f.write( ''.join(self.txtOutput) )
        f.close()


    def update(self, dt):
        """ Called once per frame
        
        Args:
            dt: int, amount of time passed since last frame.
        """
        
        # Go through events that are passed to the script by the window.
        for event in pygame.event.get():
            # We need to handle these events. Initially the only one you'll want to care
            # about is the QUIT event, because if you don't handle it, your game will crash
            # whenever someone tries to exit.
            if event.type == QUIT:
                
                self.writeOutputCsv()
                pygame.quit()
                sys.exit()      # Not including this line crashes the script on Windows. Possibly
                                # on other operating systems too, but I don't know for sure.
            
            if event.type == pygame.KEYDOWN:
                
                # Find next image to show
                if event.key == pygame.K_LEFT:
                    self.currFileInd -= 1
                    
                    # If reach the last image
                    if self.currFileInd >= len(self.fileNames):
                        self.writeOutputCsv()
                        pygame.quit()
                        sys.exit()
                    
                    if not self.fileNames[self.currFileInd].endswith('png'):
                        self.currFileInd -= 1

                if event.key == pygame.K_RIGHT:
                    self.currFileInd += 1
                    # If reach the last image
                    if self.currFileInd >= len(self.fileNames):
                        self.writeOutputCsv()
                        pygame.quit()
                        sys.exit()
                    
                    if not self.fileNames[self.currFileInd].endswith('png'):
                        self.currFileInd += 1
                
                # Appending text output
                fn = self.fileNames[self.currFileInd]
                if event.key == pygame.K_z:
                    if fn+'\n' not in self.txtOutput:
                        self.txtOutput.append(fn+'\n')
                if event.key == pygame.K_x:
                    if fn+'\n' in self.txtOutput:
                        self.txtOutput.remove(fn+'\n')
                
                # Updating image
                self.img = pygame.image.load( fn )

                # Updating text
                if self.currFileInd >= self.notFlaggedInd:
                    scriptFlag = "Script Flagged"
                else:
                    scriptFlag = "Script Not Flagged"

                if fn+'\n' in self.txtOutput:
                    flag = "Flagged"
                else:
                    flag = "Not Flagged"

                text = '%s: %s, I think %s' %(scriptFlag, self.currFileInd, flag)
                self.text = self.font.render(text, True, (0,255,0), (0,0,0))
        

    def draw(self, screen):
        """ Draw images and text to window """

        screen.fill((0, 0, 0)) # Fill the screen with black.
        screen.blit(self.img, (0, 0))
        screen.blit(self.text, (0,0))
        # Redraw screen here.
        
        # Flip the display so that the things we drew actually show up.
        pygame.display.flip()
    

    def runPyGame(self):
        """ main function """

        # Set up the clock. This will tick every frame and thus maintain a relatively constant framerate. Hopefully.
        fps = 60.0
        fpsClock = pygame.time.Clock()
        
        # Set up the window.
        pygame.display.set_caption('Eyeball segmentation')

        width, height = self.img.get_size()
        screen = pygame.display.set_mode((width, height))
        
        # screen is the surface representing the window.
        # PyGame surfaces can be thought of as screen sections that you can draw onto.
        # You can also draw surfaces onto other surfaces, rotate surfaces, and transform surfaces.
        
        
        dt = 1/fps  # dt is the time since last frame.
        while True: # Main game loop.
            
            self.update(dt) 
            self.draw(screen)
            
            dt = fpsClock.tick(fps)


if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-t',
        "--unflagged_folder",
        required=True,
        help= "path to folder with images not flagged by script")

    parser.add_argument(
        '-f',
        "--flagged_folder",
        required=True,
        help= "path to folder with images flagged by script")
    
    parser.add_argument(
        '-r',
        "--input_csv",
        default="",
        help= "path to csv file to read from")
    
    parser.add_argument(
        '-w',
        "--output_csv",
        required=True,
        help= "path to output csv file")
    
    args = parser.parse_args()
    
    game = game(args.unflagged_folder, args.flagged_folder, args.input_csv, args.output_csv)
    #game = game('C:\\Users\\User\\Desktop\\whizzstuff\\post\\small_post', 'C:\\Users\\User\\Desktop\\whizzstuff\\post\\small_flagged', 'C:\\Users\\User\\Desktop\\whizzstuff\\post\\test.txt', 'C:\\Users\\User\\Desktop\\whizzstuff\\post\\huehuehue.txt')
    game.runPyGame()
