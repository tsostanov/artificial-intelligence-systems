% =========================================================
% tests.pl — демо советчика: ДО → лог действий → ПОСЛЕ (конец хода)
% Покрываются приоритеты: PLAY-HASTE, PLAY, KILL, FACE, END_TURN
%
% Порядок загрузки в REPL:
%   ?- ['01/hearthstone.pl'].
%   ?- use_module('02/advisor.pl').
%   ?- use_module('02/tests.pl').
%
% Запуск:
%   ?- tests:demo_play.
%   ?- tests:demo_face.
%   ?- tests:demo_kill.
%   ?- tests:demo_end_turn.
%   ?- tests:demo_combo_play_kill_face.
%   ?- tests:demo_combo_play_face.
%   ?- tests:demo_all.
% =========================================================

:- encoding(utf8).

:- module(tests, [
    demo_play/0,
    demo_face/0,
    demo_kill/0,
    demo_end_turn/0,
    demo_combo_play_kill_face/0,
    demo_combo_play_face/0,
    demo_all/0
]).

:- use_module('advisor.pl').
:- use_module(library(readutil)).

% -----------------------
% ВСПОМОГАТЕЛЬНЫЕ ПРЕДИКАТЫ
% -----------------------

current_player(game_state(_,_,CP,_), CP).

% Логируем ВСЕ советы и применяем их до конца текущего хода
log_until_end(S0, SEnd) :-
    current_player(S0, CP),
    advisor:advise_now(S0),
    advisor:do_best_now(S0, S1),
    ( current_player(S1, CP) ->
        log_until_end(S1, SEnd)
    ;   SEnd = S1 ).

run_case(Title, S0) :-
    format('=== ~w: BEFORE ===~n', [Title]),
    user:display_game(S0),
    format('Action log (this turn):~n', []),
    log_until_end(S0, SEnd),
    format('=== ~w: AFTER (end of turn) ===~n', [Title]),
    user:display_game(SEnd).

pause :-
    writeln('--- Press Enter to continue ---'),
    read_line_to_string(user_input, _).

% -----------------------
% БАЗОВЫЕ СЦЕНАРИИ
% -----------------------

% PLAY: стандартное начальное состояние
scenario_play(S) :- user:initial_state(S).

% FACE: готовый 3/2, у оппонента пусто
scenario_face(game_state(
    player(30, 0, 1, [], [], [minion(bloodfen_raptor,3,2,true,false)]),
    player(30, 1, 1, [], [], []),
    1, 10)).

% KILL: мой 3/2 добивает его 1/1
scenario_kill(game_state(
    player(30, 1, 1, [], [], [minion(bloodfen_raptor,3,2,true,false)]),
    player(30, 1, 1, [], [], [minion(wisp,1,1,true,false)]),
    1, 42)).

% END_TURN: нечего играть и атак нет
scenario_end(game_state(
    player(30, 0, 1, [], [], []),
    player(30, 1, 1, [], [], []),
    1, 99)).

% -----------------------
% КОМБО-СЦЕНАРИИ (PLAY → атаки)
% -----------------------

% COMBO: PLAY (charge в руке) → KILL (3/2 по 1/1) → FACE (если останется готовый)
scenario_combo_play_kill_face(game_state(
    player(30, 1, 1, [], [card(stonetusk_boar,1,1,1)], [minion(bloodfen_raptor,3,2,true,false)]),
    player(1,  0, 1, [], [], [minion(wisp,1,1,true,false)]),
    1, 80)).

% COMBO: PLAY (charge) → FACE — летал на пустую доску
scenario_combo_play_face(game_state(
    player(30, 1, 1, [], [card(stonetusk_boar,1,1,1)], []),
    player(1,  0, 1, [], [], []),
    1, 60)).

% -----------------------
% ДЕМО
% -----------------------

demo_play                 :- scenario_play(S0),                run_case('PLAY',                S0).
demo_face                 :- scenario_face(S0),               run_case('FACE',                S0).
demo_kill                 :- scenario_kill(S0),               run_case('KILL',                S0).
demo_end_turn             :- scenario_end(S0),                run_case('END_TURN',            S0).
demo_combo_play_kill_face :- scenario_combo_play_kill_face(S0), run_case('COMBO PLAY->KILL->FACE', S0).
demo_combo_play_face      :- scenario_combo_play_face(S0),    run_case('COMBO PLAY->FACE',    S0).

demo_all :-
    writeln('Press Enter between demos...'),
    demo_play,                 pause,
    demo_face,                 pause,
    demo_kill,                 pause,
    demo_end_turn,             pause,
    demo_combo_play_kill_face, pause,
    demo_combo_play_face.
