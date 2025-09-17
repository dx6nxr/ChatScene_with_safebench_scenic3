import os
import random
import numpy as np
import torch

import sys
import time
import argparse
from collections import OrderedDict

from importlib import metadata

import scenic
import scenic.syntax.translator as translator
import scenic.core.errors as errors
# Import the necessary classes directly from Scenic
from scenic.core.simulators import SimulationCreationError, SimulationResult, TerminationType
from scenic.core.dynamics.utils import RejectSimulationException
from scenic.core.distributions import RejectionException
import scenic.core.dynamics as dynamics
from scenic.core.errors import InvalidScenarioError
from scenic.core.requirements import RequirementType
from scenic.core.vectors import Vector
# The veneer module is used for managing the simulation context
import scenic.syntax.veneer as veneer

# NOTE: The get_parser function remains the same and is omitted for brevity.
def get_parser(scenicFile):
    parser = argparse.ArgumentParser(prog='scenic', add_help=False,
                                     usage='scenic [-h | --help] [options] FILE [options]',
                                     description='Sample from a Scenic scenario, optionally '
                                                 'running dynamic simulations.')

    mainOptions = parser.add_argument_group('main options')
    mainOptions.add_argument('-S', '--simulate', default=True,
                             help='run dynamic simulations from scenes '
                                  'instead of simply showing diagrams of scenes')
    mainOptions.add_argument('-s', '--seed', help='random seed', default=0, type=int)
    mainOptions.add_argument('-v', '--verbosity', help='verbosity level (default 1)',
                             type=int, choices=(0, 1, 2, 3), default=1)
    mainOptions.add_argument('-p', '--param', help='override a global parameter',
                             nargs=2, default=[], action='append', metavar=('PARAM', 'VALUE'))
    mainOptions.add_argument('-m', '--model', help='specify a Scenic world model', default='scenic.simulators.carla.model')
    mainOptions.add_argument('--scenario', default=None,
                             help='name of scenario to run (if file contains multiple)')

    # Simulation options
    simOpts = parser.add_argument_group('dynamic simulation options')
    simOpts.add_argument('--time', help='time bound for simulations (default none)',
                         type=int, default=1)
    simOpts.add_argument('--count', help='number of successful simulations to run (default infinity)',
                         type=int, default=0)
    simOpts.add_argument('--max-sims-per-scene', type=int, default=1, metavar='N',
                         help='max # of rejected simulations before sampling a new scene (default 1)')

    # Interactive rendering options
    intOptions = parser.add_argument_group('static scene diagramming options')
    intOptions.add_argument('-d', '--delay', type=float,
                            help='loop automatically with this delay (in seconds) '
                                 'instead of waiting for the user to close the diagram')
    intOptions.add_argument('-z', '--zoom', type=float, default=1,
                            help='zoom expansion factor, or 0 to show the whole workspace (default 1)')

    # Debugging options
    debugOpts = parser.add_argument_group('debugging options')
    debugOpts.add_argument('--show-params', help='show values of global parameters',
                           action='store_true')
    debugOpts.add_argument('--show-records', help='show values of recorded expressions',
                           action='store_true')
    debugOpts.add_argument('-b', '--full-backtrace', help='show full internal backtraces',
                           action='store_true')
    debugOpts.add_argument('--pdb', action='store_true',
                           help='enter interactive debugger on errors (implies "-b")')
    debugOpts.add_argument('--pdb-on-reject', action='store_true',
                           help='enter interactive debugger on rejections (implies "-b")')
    ver = metadata.version('scenic')
    debugOpts.add_argument('--version', action='version', version=f'Scenic {ver}',
                           help='print Scenic version information and exit')
    debugOpts.add_argument('--dump-initial-python', help='dump initial translated Python',
                           action='store_true')
    debugOpts.add_argument('--dump-ast', help='dump final AST', action='store_true')
    debugOpts.add_argument('--dump-python', help='dump Python equivalent of final AST',
                           action='store_true')
    debugOpts.add_argument('--no-pruning', help='disable pruning', action='store_true')
    debugOpts.add_argument('--gather-stats', type=int, metavar='N',
                           help='collect timing statistics over this many scenes')
    
    parser.add_argument('--scenicFile', help='a Scenic file to run', default = scenicFile, metavar='FILE')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)

    # Parse arguments and set up configuration
    args = parser.parse_args(args=[])
    print (args)
    return args


class ScenicSimulator:
    def __init__(self, scenicFile, params):
        self.args = get_parser(scenicFile)
        # ... (initialization code is the same)
        delay = self.args.delay
        errors.showInternalBacktrace = self.args.full_backtrace
        if self.args.pdb:
            errors.postMortemDebugging = True
            errors.showInternalBacktrace = True
        if self.args.pdb_on_reject:
            errors.postMortemRejections = True
            errors.showInternalBacktrace = True
        translator.dumpTranslatedPython = self.args.dump_initial_python
        translator.dumpFinalAST = self.args.dump_ast
        translator.dumpASTPython = self.args.dump_python
        translator.verbosity = self.args.verbosity
        translator.usePruning = not self.args.no_pruning

        # Load scenario from file
        if self.args.verbosity >= 1:
            print('Beginning scenario construction...')
        startTime = time.time()
        
        self.scenario = errors.callBeginningScenicTrace(
            lambda: scenic.scenarioFromFile(self.args.scenicFile,
                                             params=params,
                                             model=self.args.model,
                                             scenario=self.args.scenario,
                                             mode2D=True)
        )
        self.opt_params, self.opt_record = self.get_params()
        
        totalTime = time.time() - startTime
        if self.args.verbosity >= 1:
            print(f'Scenario constructed in {totalTime:.2f} seconds.')
        self.simulator = errors.callBeginningScenicTrace(self.scenario.getSimulator)

    # Other methods like get_params, generateScene, etc., remain the same.
    def get_params(self):
        all_params = self.scenario.params
        opt_record = {}
        opt_params = {}
        for param in all_params.keys():
            if param.startswith('OPT'):
                opt_record[param] = []
                opt_params[param] = all_params[param]
                opt_params[param].min = opt_params[param].low
                opt_params[param].max = opt_params[param].high
        return opt_params, opt_record

    def record_params(self):
        if hasattr(self, 'simulation') and self.simulation:
            all_params = self.simulation.scene.params
            for param in self.opt_record.keys():
                self.opt_record[param].append(all_params[param])
            print("Recording params...")

    def update_params(self):
        print("Updating params...")
        for param in self.opt_params.keys():
            if len(self.opt_record[param]) > 1:
                mean = np.mean(self.opt_record[param])
                std = np.std(self.opt_record[param])
                self.opt_params[param].low = round(max(mean - std, self.opt_params[param].min), 2)
                self.opt_params[param].high = round(min(mean + std, self.opt_params[param].max), 2)
        self.scenario.params.update(self.opt_params)
        print(self.save_params())

    def load_params(self, params):
        print("Loading params...")
        for param in params.keys():
            self.opt_params[param].low = params[param]['low']
            self.opt_params[param].high = params[param]['high']
        self.scenario.params.update(self.opt_params)

    def save_params(self):
        print("Saving params...")
        save_params = {}
        for param in self.opt_params.keys():
            cur_param = self.opt_params[param]
            save_params[param] = {'low': cur_param.low, 'high': cur_param.high}
        return save_params

    def generateScene(self):
        scene, iterations = errors.callBeginningScenicTrace(
            lambda: self.scenario.generate(verbosity=self.args.verbosity)
        )
        return scene, iterations

    def setSimulation(self, scene):
        if self.args.verbosity >= 1:
            print(f'Creating simulation for {scene.dynamicScenario}...')
        try:
            # Use the simulator's factory method to create a simulation object
            # without running it. This is the Scenic 3.0 equivalent of the old method.
            # print (scene.objects)
            self.simulation = self.simulator.createSimulation(
                scene,
                maxSteps=self.args.time,
                timestep=0.1,
                verbosity=self.args.verbosity,
                name="Scenic_simulation"
            )
            self.simulation.client.reload_world()
        except (SimulationCreationError, RejectSimulationException) as e:
            if self.args.verbosity >= 1:
                print(f'Failed to create simulation: {e}')
            return False
        return True

    def runSimulation(self):
        """Run the simulation step-by-step, yielding at each step."""
        sim = self.simulation
        if not sim:
            raise RuntimeError("setSimulation must be called successfully first.")

        # Set up simulation context
        veneer.beginSimulation(sim)
        dynamicScenario = sim.scene.dynamicScenario
        
        # Initial setup
        try:
            # The new setup method handles object creation in the simulator
            sim.setup()
            dynamicScenario._start()
        except Exception:
            veneer.endSimulation(sim)
            raise

        # Update all objects in case the simulator adjusted properties during setup
        sim.updateObjects()
        sim.scene.requires_grad = False
        # Main simulation loop
        terminationReason = None
        terminationType = None
        actionSequence = []

        while True:
            if sim.verbosity >= 3:
                print(f'  Time step {sim.currentTime}:')
            
            # This call is now part of the public API of the Simulation object
            
            yield sim.currentTime

            # Run compose blocks and check requirements
            terminationReason = dynamicScenario._step()
            if terminationReason is not None:
                terminationType = TerminationType.scenarioComplete
                
            sim.recordCurrentState()
            
            # Run monitors
            newReason = dynamicScenario._runMonitors()
            if newReason is not None:
                terminationReason = newReason
                terminationType = TerminationType.terminatedByMonitor
            # Check for simulation termination conditions
            if terminationReason is None:
                terminationReason = dynamicScenario._checkSimulationTerminationConditions()
                if terminationReason is not None:
                    terminationType = TerminationType.simulationTerminationCondition

            # Check for timeout
            if terminationReason is None and sim.maxSteps and sim.currentTime >= sim.maxSteps:
                terminationReason = f'reached time limit ({sim.maxSteps} steps)'
                terminationType = TerminationType.timeLimit

            if terminationReason is not None:
                break
                
            # Compute agent actions
            allActions = OrderedDict()
            schedule = sim.scheduleForAgents()
            for agent in schedule:
                # The _step() call on the behavior is the main entry point.
                behavior = agent.behavior
                if not behavior._runningIterator:
                    behavior._start(agent)
                actions = behavior._step()
                
                if isinstance(actions, dynamics.EndSimulationAction):
                    terminationReason = str(actions)
                    terminationType = TerminationType.terminatedByBehavior
                    break
                if isinstance(actions, _EndScenarioAction):
                    scenario = actions.scenario
                    if scenario._isRunning:
                        scenario._stop(actions)
                    terminationReason = str(actions)
                    terminationType = TerminationType.terminatedByBehavior
                    actions = ()
                    break
                
                assert isinstance(actions, tuple)
                allActions[agent] = actions
            
            if terminationReason is not None:
                break

            # Execute actions
            actionSequence.append(allActions)
            sim.executeActions(allActions)

            # Step the physical simulation (controlled by the calling code)
            # sim.step() # As in the original, this is commented out.

            # Update Scenic objects from simulator state
            sim.updateObjects()
            sim.currentTime += 1

        # Package up results
        result = SimulationResult(
            sim.trajectory,
            actionSequence,
            terminationType,
            terminationReason,
            sim.records
        )
        sim.result = result

    def endSimulation(self):
        """Clean up after the simulation finishes or is aborted."""
        if not hasattr(self, 'simulation') or not self.simulation:
            return

        sim = self.simulation
        dynamicScenario = sim.scene.dynamicScenario

        # Evaluate final record statements
        values = dynamicScenario._evaluateRecordedExprs(RequirementType.recordFinal)
        for name, val in values.items():
            sim.records[name] = val
            
        # Stop any remaining scenarios
        for scenario in tuple(veneer.runningScenarios):
            scenario._stop('simulation terminated')

        # Clean up veneer and the simulation instance
        #sim.destroy()
        veneer.endSimulation(sim)

    def destroy(self):
        """Destroy the simulator interface."""
        if self.simulator:
            self.simulator.destroy()
