

class Cutflow:
    
    def __init__(self, output, df, cfg, processes, selection=None ):
        self.df = df
        self.cfg = cfg
        self.output = output
        self.processes = processes
        self.selection = None
        self.addRow('entry', selection)
        
    def addRow(self, name, selection, cumulative=True):
        if self.selection is None and selection is not None:
            self.selection = selection
        elif selection is not None:
            self.selection &= selection
            selection = self.selection
            
        for process in self.processes:
            if selection is not None:
                self.output[process][name] += ( sum(self.df['weight'][ (self.df['dataset']==process) & selection ].flatten() )*self.cfg['lumi'] )
                self.output[process][name+'_w2'] += ( sum((self.df['weight'][ (self.df['dataset']==process) & selection ]**2).flatten() )*self.cfg['lumi']**2 )
            else:
                self.output[process][name] += ( sum(self.df['weight'][ (self.df['dataset']==process) ].flatten() )*self.cfg['lumi'] )
                self.output[process][name+'_w2'] += ( sum((self.df['weight'][ (self.df['dataset']==process) ]**2).flatten() )*self.cfg['lumi']**2 )
  
        
