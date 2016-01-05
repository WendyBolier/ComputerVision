/*
 * MySVM.cpp
 *
 *  Created on: 27 aug. 2013
 *      Author: Coert
 */

#include "MySVM.h"

namespace nl_uu_science_gmt
{

MySVM::MySVM() :
				m_margin_location(1.000001) // we set the margin very close to 1, not actually 1 because we'll miss some support vectors
{
}

MySVM::~MySVM()
{
}

} /* namespace nl_uu_science_gmt */
