

class BaseWorkflowFactory():
    def __init__(self):

        self._wf = None

    def build(self, ):
        return NotImplementedError

    def run(self, *args,**kwargs):

        self._wf.run(*args, **kwargs)


    def show_graph(self):
        fname = self._wf.write_graph(dotfilename="graph.dot", graph2use="colored")
        return fname

def node_name(name, extra):
    if isinstance(extra, (tuple, list)):
        extra = "_".join(extra)
    return name + ("_" + extra if extra else "")

def getsubid(i):
    return f"sub_{i:02d}"

def subid_varname(i):
    return f"_sub_id_{i}"

def get_key(d, k):
    return d[k]
