"""
Example of use:

```python
def run_experiment(desc, data, clf):
    clf.fit(data.X, data.y)
    ...
    return results

def aggregate_results(results):
    ...

exp = Evaluation(run_experiment, aggregate_results)
exp.parameter('data', my_data)
exp.parameter('clf', LogisticRegression())
exp.addSearch('clf', [SVC()])

exp.runSearch()
```
"""
import collections


ParamSearch = collections.namedtuple('ParamSearch', ('name', 'alternatives', 'include_default'))


class DescribedValue:
    def __init__(self, value):
        if isinstance(value, tuple):
            self._desc, self.value = value
        else:
            self.value = value
            self._desc = None

    def __repr__(self):
        return self._desc or str(self.value)


class DescribedDict(dict):
    def __init__(self, desc, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.desc = desc

    def __repr__(self):
        return self.desc


class Parameter:
    def __init__(self, name, default_value):
        self.name = name
        self.default = DescribedValue(default_value)
        self.searches = []

    def setDefault(self, value):
        self.default = DescribedValue(value)
        return self


Search = collections.namedtuple('Search', ['name', 'alternatives', 'include_default'])


class Evaluation:
    def __init__(self, run_experiment, aggregate_results=None):
        self._run_experiment = run_experiment
        self._aggregate_results = aggregate_results
        self._params = {}
        self._searches = {}

    def parameter(self, name, default_value=None):
        """
        Defines a parameter with name `name` and optionally a default value.
        """
        if name not in self._params:
            self._params[name] = Parameter(name, default_value)

        return self._params[name]

    def addSearch(self, search_name, alternatives, include_default=True):
        """
        If `search_name` corresponds to a parameter name, `alternatives` is
        expected to be a list of alternative values for that parameter. If
        `include_default`, then the default value is included when searching
        for alternatives.

        Otherwise, `alternatives` is a list of dictionaries, in which the keys
        correspond to parameter names, and the values correspond to parameter
        values. The argument `include_default` is ignored.

        Parameters:
            - search_name: a name identifying this parameter search
            - alternatives: a list of alternatives
            - include_default: boolean
        """
        self._searches[search_name] = Search(search_name, alternatives, include_default)
        return self

    def alternatives(self, search_name):
        """
        Returns a list of dictionaries, with parameter names and values as key/value.
        """
        search = self._searches[search_name]

        if search_name in self._params:
            alternatives = [ DescribedValue(alt) for alt in search.alternatives ]
            default_value = self._params[search_name].default
            if search.include_default:
                alternatives = [default_value] + alternatives

            return [ DescribedDict(f'{alt}*' if str(alt) == str(default_value) else alt.__repr__(), {search_name: alt.value}) for alt in alternatives ]

        else:
            return search.alternatives

    def defaultValues(self):
        """
        Returns a dictionary with default parameter names and values as key/value.
        """
        return dict((param.name, param.default.value) for param in self._params.values() if param.default.value is not None)

    def getFullGrid(self, names, default_values=None):
        param_values = default_values or []

        if not names:
            yield param_values
        else:
            for alt in self.alternatives(names[0]):
                alt_params = list(param_values)
                alt_params.append((names[0], alt))

                yield from self.getFullGrid(names[1:], alt_params)

    def runFullGrid(self, names):
        """
        Runs a full grid of experiments along the dimensions in `names`.
        """
        self.runExperiments(self.getFullGrid(names))

    def getSearch(self, names=None):
        for search in self._searches.values():
            if names is None or search.name in names:
                for alt in self.alternatives(search.name):
                    yield [(search.name, alt)]

    def runSearch(self, name=None):
        """
        Runs a series of experiments along a single dimension (if `name` is not None) or all dimensions one by one.
        """
        self.runExperiments(self.getSearch(name))

    def runDefaults(self):
        self.runExperiments([[('defaults', DescribedDict('defaults'))]])

    def runExperiment(self, param_set, default_values=None):
        param_values = default_values.copy() if default_values is not None else self.defaultValues()
        for _, exp_values in param_set:
            param_values.update(exp_values)

        selected_params = [(search_name, exp_values) for search_name, exp_values in param_set]
        desc = ', '.join([f'{search_name}={exp_values}' for search_name, exp_values in param_set])

        result = self._run_experiment(**param_values, desc=desc)
        return selected_params, result

    def runExperiments(self, experiments):
        results = []
        default_values = self.defaultValues()

        for param_set in experiments:
            results.append(self.runExperiment(param_set, default_values))

        if self._aggregate_results and results:
            self._aggregate_results(results)
