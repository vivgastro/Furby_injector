from setuptools import setup

setup(  name = 'Visibility_injector',
        version = '1.0',
        packages = ['Visibility_injector',
                    'Visibility_injector.tests'],
        scripts = ['bin/plot_fake_injections.py'],
        install_requires = ['numpy', 'matplotlib', 'pyyaml', 'Furby_p3']
    )

