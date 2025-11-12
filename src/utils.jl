module SymposiumUtils

using Plots 
using Random
using Measures

#---------------------------------------------------------------------------------------------------------#
#-------------------------- MAB Environment for Generative Model Example -------------------------------#
#---------------------------------------------------------------------------------------------------------#
mutable struct MAB
    p::Float64
    switch_after::Int
    t::Int
    contingencies::Vector{Float64}
    history_obs::Vector{Int}
    history_actions::Vector{Int}
    rng::MersenneTwister
    switch_sequence::Vector{Vector{Float64}}
end

function MAB(; p=0.8, switch_after=30, seed=42)
    rng = MersenneTwister(seed)
    # define deterministic switch sequence (always same)
    switch_sequence = [
        [p, 0.5, 1-p], 
        [1-p, 0.5, p],   
        [0.5, p, 1-p],   
        [p, 1-p, 0.5]
    ]
    return MAB(p, switch_after, 0, copy(switch_sequence[1]), Int[], Int[], rng, switch_sequence)
end

function pull_arm!(env::MAB, action::Vector{Int})
    action = action[1]  # extract action index
    env.t += 1

    # determine reward/loss
    prob = env.contingencies[action]
    obs = rand(env.rng) < prob ? 1 : 2

    # save to history
    push!(env.history_actions, action)
    push!(env.history_obs, obs)

    # switch contingencies if switch_after reached
    if env.t % env.switch_after == 0
        idx = ((env.t ÷ env.switch_after) % length(env.switch_sequence)) + 1
        env.contingencies = copy(env.switch_sequence[idx])
    end

    return [obs]
end

function plot_history(env::MAB, store_A_matrices::Vector{Vector{Float64}}, store_posterior_states::Vector{Vector{Float64}})
    n_arms = length(env.contingencies)
    n_trials = env.t

    # --------------------------
    # (1) Environment Heatmap
    # --------------------------
    prob_matrix = zeros(Float64, n_arms, n_trials)
    for trial in 1:n_trials
        switch_idx = ((trial - 1) ÷ env.switch_after) % length(env.switch_sequence) + 1
        prob_matrix[:, trial] = env.switch_sequence[switch_idx]
    end

    p1 = heatmap(
        1:n_trials, 1:n_arms, prob_matrix;
        yflip = true,
        yticks = (1:n_arms),
        color = cgrad(:Purples),
        clims = (0.0, 1.0),
        xlabel = "",
        ylabel = "Arm",
        title = "MAB Reward Probabilities and Actions",
        cbar_title = "P(Reward)",
        xticks = (0:25:100),        # <-- added for alignment
        xlims = (0, 100),           # <-- force same horizontal range
        size = (800, 250),
        legend = true
    )

    for (trial, action, obs) in zip(1:n_trials, env.history_actions, env.history_obs)
        scatter!(p1, [trial], [action],
                 color = (obs == 1 ? :green : :red),
                 markersize = 4, legend = false)
    end

    # --------------------------
    # (2) A-Matrix Evolution
    # --------------------------
    A_mat = hcat(store_A_matrices...)
    p2 = heatmap(
        1:size(A_mat, 2), 1:size(A_mat, 1), A_mat;
        yflip = true,
        yticks = (1:size(A_mat, 1)),
        xticks = (0:25:100),
        xlims = (0, 100),           # ensure same x range
        xlabel = "",
        ylabel = "Arm",
        cbar_title = "P(o|s)",
        title = "A-Matrix",
        color = cgrad(:Purples),
        size = (800, 200)
    )

    # --------------------------
    # (3) Posterior over States
    # --------------------------
    post_mat = hcat(store_posterior_states...)
    p3 = heatmap(
        1:size(post_mat, 2), 1:size(post_mat, 1), post_mat;
        yflip = true,
        yticks = (1:size(post_mat, 1)),
        xticks = (0:25:100),
        xlims = (0, 100),           # ensure same x range
        xlabel = "Trial",
        ylabel = "State",
        cbar_title = "P(s|o)",
        title = "Posterior over States",
        color = :BuGn,
        size = (800, 200)
    )

    # --------------------------
    # Combine vertically
    # --------------------------
    plot(p1, p2, p3; layout = @layout([a; b; c]), size = (1000, 550),
     left_margin = 10mm,
     bottom_margin = 5mm)
end






end