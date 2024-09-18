for /D %%p IN ("Results\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

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
del /F /Q Input\EE\Test\SplitMetaData\*
for /D %%p IN ("Input\PD\Training\EventLogs\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
for /D %%p IN ("Input\CC\Training\EventLogs\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
for /D %%p IN ("Input\CC\Training\PetriNets\*") DO (
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