// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <chrono>
#include <ctime>

#include <doctest/doctest.h>
#include <interpreter/dispatch_table.hpp>
#include <thread>

#include "core/dataset.hpp"
#include "core/pset.hpp"
#include "core/version.hpp"
#include "interpreter/interpreter.hpp"
#include "nanobench.h"
#include "operators/creator.hpp"
#include "operators/evaluator.hpp"
#include "parser/infix.hpp"
#include <fstream>
#include <iostream>
#include <string.h>
#if TF_MINOR_VERSION > 2
#include <taskflow/algorithm/reduce.hpp>
#endif
#include <taskflow/taskflow.hpp>

namespace Operon {
namespace Test {

    namespace nb = ankerl::nanobench;

    void get_results(std::string const& function_set, 
        std::string const& fitness_case, int progs_per_bin, 
        int num_size_bins, std::ofstream& out_file)
    {
        /* Calculate some profiling results. */

        // Number of times that nanobench is independently executed,
        // in order to generate a list of median average runtimes.
        const int NB_NUM_GENERATIONS = 11;

        // Number of epochs within a single nanobench run.
        const int NB_NUM_EPOCHS = 1;

        // Number of iterations within a single nanobench epoch.
        const int NB_NUM_ITERATIONS = 1;

        // File path to the relevant program strings.
        std::string prog_path = "../../../../results/programs/" + 
            function_set + "/programs_operon.txt";

        // File path to the relevant dataset.
        std::string fit_path = "../../../../results/programs/" + 
            function_set + "/fitness_cases/" + fitness_case;

        // Object to contain the relevant dataset.
        auto ds = Dataset(fit_path, true);

        // used for parsing and printing
        robin_hood::unordered_flat_map<std::string, Operon::Hash> vars_map;
        std::unordered_map<Operon::Hash, std::string> vars_names;
        for (auto const& v : ds.Variables()) {
            vars_map[v.Name] = v.Hash;
            vars_names[v.Hash] = v.Name;
        }

        // Name for target vector.
        auto target = "y";

        // Retrieve fitness cases for the relevant variable terminals.
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(
            variables.begin(), variables.end(),
            std::back_inserter(inputs), [&](auto const& v) { 
                return v.Name != target; });
        Range range = { 0, ds.Rows() };

        auto problem = Problem(ds).Inputs(inputs).Target(target).
            TrainingRange(range).TestRange(range);

        // File containing program strings.
        std::ifstream file(prog_path);

        std::string str;

        tf::Executor executor(std::thread::hardware_concurrency());
        std::vector<Operon::Vector<Operon::Scalar>> slots(
            executor.num_workers());
        for (auto& s : slots) { s.resize(range.Size()); }

        Operon::Interpreter interpreter;
        Operon::Evaluator<Operon::RMSE, false> evaluator(problem, interpreter);

        for (int bin = 0; bin < num_size_bins; bin++) {
            // For size bin `bin`...
            auto time_ = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(time_);
            std::string t(std::ctime(&time));
            t = t.substr(0, t.length() - 1);
            std::cout << "(" << t << ") "
                      << "Size bin `" << bin + 1 << "`...\n";

            // Vector to contain program strings for size bin `i`.
            std::vector<Individual> individuals(progs_per_bin);

            // Retrieve the relevant program strings for size bin `i`.
            for (int i = 0; i < progs_per_bin; i++) {
                if (std::getline(file, str)) {
                    individuals[i].Genotype = Operon::InfixParser::Parse(
                        str, vars_map);
                    ENSURE(individuals[i].Genotype.Length() > 0);
                }
            }

            nb::Bench outer_b;

            auto totalNodes = std::transform_reduce(
                individuals.cbegin(), individuals.cend(), 0UL, std::plus<>{}, 
                    [](auto const& individual) { 
                        return individual.Genotype.Length(); });
            Operon::Vector<Operon::Scalar> buf(range.Size());
            Operon::RandomGenerator rd(1234);


            // Nanobench evaluator.
            auto test = [&](nb::Bench& b, std::string const& name, 
                int epochs, int epoch_iterations) {
                evaluator.SetLocalOptimizationIterations(0);
                evaluator.SetBudget(std::numeric_limits<size_t>::max());

                b.batch(totalNodes * range.Size()).epochs(epochs).
                    epochIterations(epoch_iterations).run(name, [&]() {
                    tf::Taskflow taskflow;
                    double sum { 0 };
                    taskflow.transform_reduce(individuals.begin(), 
                        individuals.end(), sum, std::plus<> {},
                            [&](Operon::Individual& ind) {
                        auto id = executor.this_worker_id();
                        auto f = evaluator(rd, ind, slots[id]).front();

                        // printf("Fitness: %f\n", f);

                        return f;
                    });
                    executor.run(taskflow).wait();
                    return sum;
                });
            };

            int num_generations = NB_NUM_GENERATIONS;
            int num_epochs = NB_NUM_EPOCHS;
            int num_iterations = NB_NUM_ITERATIONS;

            for (int gen = 0; gen < num_generations; gen++) {
                test(outer_b, "RMSE", num_epochs, num_iterations);
            }

            // For the relevant size bin, print out a list of
            // `NB_NUM_GENERATIONS` median runtimes calculated
            // by running nanobench `NB_NUM_GENERATIONS` times.
            // For each nanobench run, there were `NB_NUM_EPOCHS`
            // epochs, and for each epoch, there were `NB_NUM_
            // ITERATIONS` iterations.

            std::string str_bin = std::to_string(bin);
            out_file << "bin: " + str_bin + ",";
            auto results = outer_b.results();

            for (int i = 0; i < NB_NUM_GENERATIONS; i++) {
                // Retrieve median runtime for each "generation".
                double median = results[i].median(
                    nb::Result::Measure::elapsed);

                // Write median runtime in terms of microseconds,
                // to utilize more significant digits.
                out_file << std::to_string(median * 1000000);

                if (NB_NUM_GENERATIONS - i != 1)
                    out_file << ",";
            }

            out_file << "\n";
        }
    }

    TEST_CASE("Node Evaluations Batch")
    {
        const int NUM_FITNESS_CASE_AMOUNTS = 5;

        const int NUM_FUNCTION_SETS = 3;

        const int NUM_PROGRAMS_PER_BIN = 1024;

        std::string fitness_cases[NUM_FITNESS_CASE_AMOUNTS] = { 
            // "1000.csv" };
            "10.csv", "100.csv", "1000.csv", "10000.csv", "100000.csv" };

        std::string fitness_cases_names[NUM_FITNESS_CASE_AMOUNTS] = { 
            // "1000" };
            "10", "100", "1000", "10000", "100000" };

        std::string function_sets[NUM_FUNCTION_SETS] = { 
            "nicolau_a", "nicolau_b", "nicolau_c" };

        int size_bins[NUM_FUNCTION_SETS] = { 32, 32, 32 };

        std::cout << "\n\nOperon build information: " << 
            Operon::Version() << "\n\n";

        for (int i = 0; i < NUM_FUNCTION_SETS; i++) {
            // For each function set...

            std::ofstream out_file;
            out_file.open(
                "../../../../results/" + std::string("results_operon_") + 
                    function_sets[i] + ".csv");
            
            for (int j = 0; j < NUM_FITNESS_CASE_AMOUNTS; j++) {
                // For each number of fitness cases...
                std::cout << "\nNumber of fitness cases: " << 
                    fitness_cases_names[j] << "\n\n";

                get_results(
                    function_sets[i], fitness_cases[j],
                    NUM_PROGRAMS_PER_BIN, size_bins[i], out_file);

            }

            out_file.close();
        }
    }

} // namespace Test
} // namespace Operon