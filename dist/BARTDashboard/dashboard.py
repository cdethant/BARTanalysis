import os
import sys
import tempfile
import shutil
import atexit
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import drive_reader
import batch_analysisBART as batch
import indiv_analysisBART as indiv

'''No longer using PHP framework: Uncomment if implementing on a server (where PHP would make more sense)'''
'''
def update_PHP(endpoint):
    try:
        response = requests.get(endpoint)
        response.raise_for_status()

        result = response.json()

        if result['success']:
            return {
                'success': True,
                'fileId': result['fileId'],
                'fileName': result['fileName']
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error occurred')
            }
            
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Request failed: {str(e)}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Error: {str(e)}"
        }
    
def fetch_trials():
    # Makes a get request to drive_reader.php to fetch trials from drive
    # url of the php endpoint
    url = "http://localhost:8000/drive_reader.php"

    # Make Get Request to call PHP script
    response = requests.get(url)

    # Check if request was successful
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")

    # PHP script returns a dict and python code to execute
    # We can execute the text below:
    python_code = response.text

    fetched = {}
    exec(python_code, {'pd': pd}, fetched)

    return fetched['files']'
    '''

def cleanup_figures():
    '''Clean up temp files when run in .exe deployment'''
    temp_figures = os.path.join(tempfile.gettempdir(), "figures")
    if getattr(sys, 'frozen', False):
        temp_figures = os.path.join(tempfile.gettempdir(), "figures")
        if os.path.exists(temp_figures):
            try:
                shutil.rmtree(temp_figures)
                print("Removed figures directory in temp path")
            except Exception as e:
                print(f"Error removing figures directory in temp: {str(e)}")

    else:
        pass

# Register the cleanup function to be called when the program exits
atexit.register(cleanup_figures)

class SelectTrials:
    def __init__(self,root):
        self.root = root
        self.root.title("BART Online Dashboard")

        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Calculate window size to fit within screen with margins
        # Use a percentage of screen size, but leave margin for taskbar and borders
        window_width = min(850, int(screen_width * 0.9))
        window_height = min(1150, int(screen_height * 0.9))
        
        # Calculate position coordinates for center of screen
        center_x = int((screen_width - window_width) / 2)
        center_y = int((screen_height - window_height) / 2)
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        
        # Calculate appropriate figure size based on window size
        # Reserve space for UI elements (approximately 250px)
        fig_height = window_height - 250
        # Width should maintain aspect ratio
        fig_width = int(fig_height * 0.89)
        
        # Store figure dimensions for later use
        self.fig_width = fig_width
        self.fig_height = fig_height
        
        # Allow window to be resized proportionally
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Store the taskdf reference
        self.taskdf = taskdf
        self.trials = trials
        self.current_trial_index = 0  # Default to first trial

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        # Top controls frame
        top_frame = ttk.Frame(self.main_frame)
        top_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=2)
        
        # Dropdown for individual trial selection
        ttk.Label(top_frame, text="Select Trial:").grid(row=0, column=0, sticky=tk.W, padx=(0,5))
        self.type_var = tk.StringVar()
        self.type_dropdown = ttk.Combobox(top_frame, textvariable=self.type_var, width=30)
        self.type_dropdown['values'] = trials
        self.type_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.type_dropdown.set(trials[0] if trials else 'No trials available')

        # Bind the dropdown selection event
        self.type_dropdown.bind("<<ComboboxSelected>>", self.on_trial_selected)

        # Status label
        self.status_label = ttk.Label(self.main_frame, text="")
        self.status_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Radio button frame
        radio_frame = ttk.LabelFrame(self.main_frame, text="View Mode")
        radio_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Radiobuttons
        self.view = tk.BooleanVar(value=True)
        ttk.Radiobutton(radio_frame, 
                text="Individual Analysis", 
                variable=self.view,
                value=False,
                width=20,
                command=self.update_figure
                ).grid(row=0, column=0, sticky=tk.W, padx=20, pady=10)

        ttk.Radiobutton(radio_frame, 
                text="Batch Analysis", 
                variable=self.view,
                value=True,
                width=20,
                command=self.update_figure
                ).grid(row=0, column=1, sticky=tk.W, padx=20, pady=10)
        
        # Create a dictionary to store generated figures
        self.figures = {
            'batch': {},  # Will store batch figures for each trial
            'indiv': {}   # Will store individual figures for each trial
        }
        
        # Create image frame with border
        img_frame = ttk.LabelFrame(self.main_frame, text="Analysis Figure")
        img_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.N, tk.S, tk.E, tk.W))
        img_frame.rowconfigure(0, weight=1)
        img_frame.columnconfigure(0, weight=1)
        
        # Create label for image
        self.img_label = ttk.Label(img_frame)
        self.img_label.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Generate initial figures
        self.generate_figures()
        
        # Button frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Generate Figure button
        ttk.Button(button_frame, text="Generate New Analysis Figures", 
                  command=self.generate_figures).pack(fill=tk.X, padx=10)
        
        # Display initial figure
        self.update_figure()
    
    def on_trial_selected(self, event):
        """Handle trial selection from dropdown"""
        selected_trial = self.type_var.get()
        if selected_trial in self.trials:
            self.current_trial_index = self.trials.index(selected_trial)
            self.update_figure()
            self.status_label.config(text=f"Selected: {selected_trial}")

    def generate_figures(self):
        """Generate and store figures for current trial"""
        selected_trial = self.type_var.get()
        
        self.status_label.config(text=f"Generating figures for {selected_trial}...")
        print("Generating figures...")
        self.root.update()
        
        try:
            # Generate batch figure if not already generated
            if selected_trial not in self.figures['batch']:
                # Run batch analysis and save figure
                _, batch_path = batch.main(self.taskdf)
                
                # Load and store the figure
                batch_img = Image.open(batch_path)
                batch_img = batch_img.resize((self.fig_width, self.fig_height), Image.Resampling.LANCZOS)
                self.figures['batch'][selected_trial] = ImageTk.PhotoImage(batch_img)
            
            # Generate individual figure if not already generated
            if selected_trial not in self.figures['indiv']:
                # Run individual analysis and save figure
                _, indiv_path = indiv.main(self.taskdf[selected_trial], selected_trial)
                
                # Load and store the figure
                indiv_img = Image.open(indiv_path)
                indiv_img = indiv_img.resize((self.fig_width, self.fig_height), Image.Resampling.LANCZOS)
                self.figures['indiv'][selected_trial] = ImageTk.PhotoImage(indiv_img)
            
            self.status_label.config(text=f"Figures generated for {selected_trial}")
        except Exception as e:
            self.status_label.config(text=f"Error generating figures: {str(e)}")

    def update_figure(self):
        """Update UI displayed figure based on current selection fromn dropdown"""
        
        selected_trial = self.type_var.get()
        
        # Check if we need to generate figures first
        if (self.view.get() and selected_trial not in self.figures['batch']) or \
           (not self.view.get() and selected_trial not in self.figures['indiv']):
            self.generate_figures()
        
        # Display appropriate figure
        if self.view.get():  # Batch view
            if selected_trial in self.figures['batch']:
                self.img_label.configure(image=self.figures['batch'][selected_trial])
            else:
                self.status_label.config(text="Batch figure not available")
        else:  # Individual view
            if selected_trial in self.figures['indiv']:
                self.img_label.configure(image=self.figures['indiv'][selected_trial])
            else:
                self.status_label.config(text="Individual figure not available")


# Call Python script to get online files
taskdf = drive_reader.drive_reader("1NMptnElGa-4xA42bAZ6VOCbRCCibw20o")

trials = []

for key, value in taskdf.items() :
    trials.append(key)

def main():
    root = tk.Tk()
    root.title("Dashboard")
    app = SelectTrials(root)

    # Add explicit call to run the cleanup when the window is closed
    root.protocol("WM_DELETE_WINDOW", lambda: (root.destroy(), cleanup_figures()))

    root.mainloop()

if __name__=="__main__":
    main()