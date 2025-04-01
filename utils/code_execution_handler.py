# SPDX-License-Identifier: MIT

def local_code_execution(completion, unit_tests, timeout=3):
    output_dict = {
        "correct_tests": [],
        "average_test_score": 0.0,
        "unit_test_stdouts": [],
        "unit_test_stderrs": [],
        "traceback": [],
        "time_taken": [],
    }

    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        custom_globals = {"__builtins__": __builtins__}

        for _, inp in enumerate(unit_tests):
            start = time.time()

            custom_globals = {"__builtins__": __builtins__}
            try:
                with time_limit(timeout):
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    exec(completion + inp, custom_globals)

                err = sys.stderr.getvalue()
                output_dict['correct_tests'].append(err == '')
                output_dict['unit_test_stderrs'].append(err)
                output_dict['traceback'].append("\n".join(traceback.format_exc().split("\n")[3:]))

            except Exception as e:
                output_dict['correct_tests'].append(False)
                output_dict['unit_test_stderrs'].append(repr(e))
                output_dict['traceback'].append("\n".join(traceback.format_exc().split("\n")[3:]))

            finally:
                out = sys.stdout.getvalue()
                err = sys.stderr.getvalue()
                output_dict['time_taken'].append(time.time() - start)
                output_dict['unit_test_stdouts'].append(out)

        output_dict['average_test_score'] = (
            0.0
            if len(output_dict['correct_tests']) == 0
            else (sum(output_dict['correct_tests']) / len(output_dict['correct_tests']))
        )

        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

        return output_dict

