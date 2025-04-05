#include "SJLog.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
LOG_LEVEL SJLog::m_level = LOG_LEVEL::NOTHING;
char* strrev1(char* str)
{
	char* end, * wrk = str;
	{
		if (str && *str)
		{
			end = str + strlen(str) - 1;
			while (end > wrk)
			{
				char temp;

				temp = *wrk;
				*wrk++ = *end;
				*end-- = temp;
			}
		}
	}
	return str;
}
SJLog::SJLog()
{
	//	m_level = LOG_LEVEL::NOTHING;
}
SJLog::~SJLog()
{

}
void SJLog::SetLOGLevel(const LOG_LEVEL level)
{
	m_level = level;
}
void SJLog::PrintLOG(const LOG_LEVEL level, const char* file, const char* func, const int line_num, const char* format, ...)
{
	if (m_level >= level) {
		char copyPath[1024];
		char filename[1024];
		char* prefix;
		strcpy(copyPath, file);
		//strcpy_s(copyPath, sizeof(copyPath), file);
		prefix = strrev1(copyPath);

		//char* context = NULL;
		//char* token = strtok_s(prefix, "\\", &context);
		//char* token1 = strtok_s(token, ".", &context);
		//prefix = _strrev(context);

		char* token = strtok(prefix, "/");
		char* token1 = strtok(token, ".");
		prefix = strrev1(strtok(NULL, "."));

		va_list ap;
		va_start(ap, format);
		switch (level) {
		case LOG_LEVEL::INFO:
			printf("Info : class : %s | function : %s | line : %d | ", prefix, func, line_num);
			break;
		case LOG_LEVEL::WARNING:
			printf("Warning : class : %s | function : %s | line : %d | ", prefix, func, line_num);
			break;
		case LOG_LEVEL::DETAIL:
			printf("Detail : class : %s | function : %s | line : %d | ", prefix, func, line_num);
			break;
		case LOG_LEVEL::VERBOSE:
			printf("Verbose : class : %s | function : %s | line : %d | ", prefix, func, line_num);
			break;
		}
		vprintf(format, ap);
		va_end(ap);
	}
}