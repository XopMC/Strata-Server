#pragma once
#include "whole_operator_v303.hpp"
namespace ds303 {
struct Plan {
 uint32_t rows=0,routes=0,gu_words=0,down_words=0;
 uint32_t gate[MaxRoutes]{},up[MaxRoutes]{},down[MaxRoutes]{},original_slot[MaxRoutes]{};
 float alpha[MaxRoutes]{};
};
static_assert(sizeof(Plan)==736);
void gather303(const Plan*,const float*raw,float*gate,float*up,cudaStream_t);
void scatter_bf16303(const Plan*,const float*raw,uint16_t*out,int*error,cudaStream_t);
}
