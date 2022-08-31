

class BaseWorkflowFactory():
    def __init__(self):

        self._wf = None

    def build(self, ):
        return NotImplementedError


    @staticmethod
    def run(wf, iter_on=None, plugin=None):
        if iter_on is not None:
            wf.get_node("infosource").iterables = iter_on
        wf.run(plugin)

    @staticmethod
    def show_graph(wf):
        fname = wf.write_graph(dotfilename="graph.dot", graph2use="colored")
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
