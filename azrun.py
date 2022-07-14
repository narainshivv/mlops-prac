import azureml
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import Workspace, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.exceptions import ComputeTargetException
from azureml.core.runconfig import DockerConfiguration
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import ScriptRunConfig
# from azureml.core.model import Model

ia = InteractiveLoginAuthentication(tenant_id='e4e34038-ea1f-4882-b6e8-ccd776459ca0')
ws = Workspace.from_config(auth=ia)
print(f'worspace details {ws}')
# def registerModel(model_path, model_name):
# 	Model.register(workspace=ws, model_path=model_path, model_name=model_name)
cluster_name = 'cluster-diabetes'

try:
	compute_target = ComputeTarget(workspace=ws, name=cluster_name)
	print('Found existing coumpute target')
except ComputeTargetException:
	print('Createing a new compute target...')
	compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_D2as_v4', max_nodes=2)
	compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
	compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

print('-'*101)
print('compute target created')
print('-'*101)

keras_env = Environment(workspace=ws, name='diabetes-env')

print('loading the conda dependencies..')
for pip_package in ["joblib","scikit-learn","pandas","azureml-sdk", "numpy", "argparse"]:
    keras_env.python.conda_dependencies.add_pip_package(pip_package)

args = ['--reg_rate', 0.1]
print('-'*101)
print('Environment variables created...')
print('-'*101)
docker_config = DockerConfiguration(use_docker=True)
print('running the script my friend ....')
src = ScriptRunConfig(source_directory='.',
						script='train.py',
						arguments=args,
						compute_target=compute_target,
						environment=keras_env,
						docker_runtime_config= docker_config
						)
print('completed running the script...')
run = Experiment(workspace=ws, name='diabetes-workspace').submit(src)
run.wait_for_completion(show_output=True)

model_name = 'diabetes_model.pkl'

print('registering the model...')
if run.get_status() == 'Completed':
	model = run.register_model(
    		model_name=model_name,
        	model_path=f'outputs/{model_name}'
        )
	print('model registered!')

print('Experiment completed ..............')