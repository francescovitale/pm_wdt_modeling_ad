:: Options:
:: split_type=[per_sensor,per_sensor_type]
:: split_fraction=<integer>
:: pd_variant=[im,imf_20,alpha,heuristics,ilp]
:: cc_variant=[token_based, alignment_based]
:: cc_variant_type=[dijkstra_less_memory, dijkstra_no_heuristics, discounted_a_star, state_equation_a_star, tweaked_state_equation_a_star]

:: %1: split type, %2: split fraction, %3: pd variant, %4: cc_variant

set split_type=per_sensor_type
set split_fraction=3
set pd_variant=heuristics
set cc_variant=alignment_based
set cc_variant_type=state_equation_a_star
set n_reps=1

for /D %%p IN ("Results\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for /l %%a in (1, 1, %n_reps%) do (


	for /D %%p IN ("Input\EE\Training\Data\*") DO (
		del /s /f /q %%p\*.*
		for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
		rmdir "%%p" /s /q
	)

	for /D %%p IN ("Input\EE\Test\Data\*") DO (
		del /s /f /q %%p\*.*
		for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
		rmdir "%%p" /s /q
	)
	
	cd WDTData
	
	call autorun
	
	cd ..
	
	xcopy WDTData\Output\Training\Data Input\EE\Training\Data /E
	xcopy WDTData\Output\Test\Data Input\EE\Test\Data /E
	for /D %%p IN ("Input\EE\Test\Data\N\*") DO (
		del /s /f /q %%p\*.*
		for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
		rmdir "%%p" /s /q
	)
	rmdir Input\EE\Test\Data\N /s /q
	ren Input\EE\Test\Data\N_modeling N

	for %%x in (%pd_variant%) do (

		mkdir Results\%%x

		for %%y in (%split_fraction%) do (
		
			for /D %%p IN ("Output\EE\Training\EventLogs\*") DO (
				del /s /f /q %%p\*.*
				for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				rmdir "%%p" /s /q
			)
			del /F /Q Output\EE\Training\SplitMetaData\*
			for /D %%p IN ("Output\EE\Test\EventLogs\*") DO (
				del /s /f /q %%p\*.*
				for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				rmdir "%%p" /s /q
			)
			for /D %%p IN ("Output\PD\Training\PetriNets\*") DO (
				del /s /f /q %%p\*.*
				for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				rmdir "%%p" /s /q
			)
			del /F /Q Output\CC\Test\Metrics\*
			del /F /Q Output\CC\Test\Diagnoses\*

			del /F /Q Input\EE\Test\SplitMetaData\*
			for /D %%p IN ("Input\PD\Training\EventLogs\*") DO (
				del /s /f /q %%p\*.*
				for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				rmdir "%%p" /s /q
			)
			for /D %%p IN ("Input\CC\Test\EventLogs\*") DO (
				del /s /f /q %%p\*.*
				for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				rmdir "%%p" /s /q
			)
			for /D %%p IN ("Input\CC\Test\PetriNets\*") DO (
				del /s /f /q %%p\*.*
				for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				rmdir "%%p" /s /q
			)

			python event_extraction.py Training %split_type% %%y

			xcopy Output\EE\Training\EventLogs Input\PD\Training\EventLogs /E
			copy Output\EE\Training\SplitMetaData\* Input\EE\Test\SplitMetaData

			python event_extraction.py Test %split_type% %%y

			python process_discovery.py %%x

			copy Output\PD\Training\PetriNets\N\Tank.pnml Results\%%x
			ren Results\%%x\Tank.pnml Tank_%%y_%%a.pnml
			copy Output\PD\Training\Metrics\statistics.txt Results\%%x
			ren Results\%%x\statistics.txt pd_statistics_%%y_%%a.txt

			xcopy Output\PD\Training\PetriNets Input\CC\Test\PetriNets /E
			xcopy Output\EE\Test\EventLogs Input\CC\Test\EventLogs /E

			python conformance_checking.py Test %cc_variant% %cc_variant_type% N_only Modeling
			
			copy Output\CC\Test\Metrics\time.txt Results\%%x
			ren Results\%%x\time.txt metrics_time_%%y_%%a.txt
			copy Output\CC\Test\Metrics\fitness.txt Results\%%x
			ren Results\%%x\fitness.txt fitness_%%y_%%a.txt
			copy Output\CC\Test\Metrics\precision.txt Results\%%x
			ren Results\%%x\precision.txt precision_%%y_%%a.txt
			copy Output\CC\Test\Metrics\generalization.txt Results\%%x
			ren Results\%%x\generalization.txt generalization_%%y_%%a.txt
			copy Output\CC\Test\Metrics\simplicity.txt Results\%%x
			ren Results\%%x\simplicity.txt simplicity_%%y_%%a.txt
		)
	)
)