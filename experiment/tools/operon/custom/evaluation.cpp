// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <cstdio>
#include <ctime>
#include <chrono>

#include <doctest/doctest.h>
#include <interpreter/dispatch_table.hpp>
#include <thread>

#include "core/dataset.hpp"
#include "core/version.hpp"
#include "interpreter/interpreter.hpp"
#include "core/pset.hpp"
#include "operators/creator.hpp"
#include "operators/evaluator.hpp"
#include "parser/infix.hpp"
#include "nanobench.h"
#include <fstream>
#include <iostream>
#include "taskflow/taskflow.hpp"
#include <string.h>

namespace Operon {
namespace Test {

    namespace nb = ankerl::nanobench;

    void get_results(std::string const& primitive_set, 
            int n_fitness_cases, int n_bins, int n_programs, 
            int n_runs, std::ofstream& results_file)
        
        {
        /* Calculate some profiling results. */

        // File path to the relevant program strings.
        std::string program_file_path = "../../../../results/programs/" + 
            primitive_set + "/programs_operon.txt";

        // File path to the relevant dataset.
        std::string data_file_path = "../../../../results/programs/" + 
            primitive_set  + "/" + std::to_string(n_fitness_cases) + 
            "/data.csv"; 

        // Object to contain the relevant dataset.
        auto ds = Dataset(data_file_path, true);

        robin_hood::unordered_flat_map<std::string, Operon::Hash> vars_map;
        for (auto const& v : ds.Variables()) {
            vars_map[v.Name] = v.Hash;
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
        std::ifstream file(program_file_path);

        std::string str;

        // Create Taskflow executor with maximal number of threads.
        // tf::Executor executor(1);
        tf::Executor executor(std::thread::hardware_concurrency());
        std::vector<Operon::Vector<Operon::Scalar>> slots(
            executor.num_workers());
        for (auto& s : slots) { s.resize(range.Size()); }

        Operon::Interpreter interpreter;
        Operon::Evaluator<Operon::RMSE, false> evaluator(problem, interpreter);

        for(int bin = 0; bin < n_bins; bin++)
        {
            // Vector to contain program strings for bin.
            std::vector<Individual> individuals(n_programs);

            // Retrieve the relevant program strings for the bin.
            for (int i = 0; i < n_programs; i++) {
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

            auto experiment = [&](nb::Bench& b, std::string const& name, 
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
                        auto fitness = evaluator(rd, ind, slots[id]).front();

                        // std::cout << "\nFitness: " << fitness;

                        return fitness;
                    });
                    executor.run(taskflow).wait();
                    return sum;
                });
            };

            const int n_epochs = 1;
            const int n_iterations = 1;

            auto time_ = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(time_);
            std::string t(std::ctime(&time));
            t = t.substr(0, t.length() - 1);
            std::cout << "\n\n(" << t << ") " << "Operon: evaluating " <<
                "programs for primitive set `" << primitive_set << "`, bin " 
                << bin + 1 << ", " << n_fitness_cases << " fitness " <<
                "cases...\n\n";

            for (int gen = 0; gen < n_runs; gen++) {
                experiment(outer_b, "RMSE", n_epochs, n_iterations);
            }
            
            // For the relevant program bin, print out a list of
            // `n_runs` runtimes calculated by running nanobench 
            // experiment `n_runs` times.
            auto results = outer_b.results();
            for (int i = 0; i < n_runs; i++)
            {
                // Retrieve median runtime for each "generation".
                double median = results[i].median(nb::Result::Measure::elapsed);

                // Write median runtime in terms of microseconds,
                // to utilize more significant digits.
                results_file << std::to_string(median*1000000);

                if (n_runs - i != 1)
                    results_file << ",";
            }

            results_file <<"\n";
            
        }

        return;
    }

    TEST_CASE("Node Evaluations Batch")
    {
        const int n_primitive_sets = 3;

        std::string primitive_sets[n_primitive_sets] = 
            {"nicolau_a", "nicolau_b", "nicolau_c"};

        const int n_fitness_case_amounts = 5;

        int n_fitness_cases[n_fitness_case_amounts] = {
            10, 100, 1000, 10000, 100000};
            // 10, 100};

        const int n_bins = 32;

        const int n_programs = 512;

        const int n_runs = 11;

        std::cout << "\n\nOperon build information: " << Operon::Version() <<
            "\n\n";

        for(int i = 0; i < n_primitive_sets; i++)
        {
            // For each function set...

            std::ofstream results_file;
            results_file.open("../../../../results/runtimes/operon/" + 
                primitive_sets[i] + ".csv");
            
            for(int j = 0; j < n_fitness_case_amounts; j++)
            {
                get_results(primitive_sets[i], n_fitness_cases[j],
                    n_bins, n_programs, n_runs, results_file);    
            }

            results_file.close();
            results_file.clear();

        }

    }

} // namespace Test
} // namespace Operon