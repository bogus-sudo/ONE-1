/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "program_options.h"

#include <stdexcept>

Option::Option(const std::string& name): its_name_(name) {}

const std::string& Option::name() const noexcept { return its_name_; }

Option& Option::description(const std::string& its_description) {
   its_description_ = its_description;
   return *this;
}

const std::string& Option::description() const noexcept { return its_description_; }

void Option::value(const std::string& its_value) const {
   its_value_ = its_value;
}

const std::string& Option::value() const noexcept { return its_value_; }

static bool operator<(const Option &lhs, const Option &rhs) { return lhs.name() < rhs.name(); }


ProgramOptions::ProgramOptions(const char* args[], size_t number_of_args)
   : command_line_arguments_(args)
   , number_of_command_line_arguments_(number_of_args)
   , need_to_show_user_guide_(false)
{
   for (size_t arg_num = 1; arg_num < number_of_args; ++arg_num) {
      std::string command_line_key(args[arg_num]);
      if (command_line_key == "--help" || command_line_key == "-h") {
         need_to_show_user_guide_ = true;
      }
   }
}

ProgramOptions& ProgramOptions::define(const Option& option) {
   option.value(getFromCommandLineArgumentsValueOf(option));
   mandatory_options_.insert(option);
   return *this;
}

std::string ProgramOptions::getFromCommandLineArgumentsValueOf(const Option &option) const {
   for (size_t argument_idx = 1; argument_idx < number_of_command_line_arguments_-1; ++argument_idx) {
      auto command_line_argument = command_line_arguments_[argument_idx];
      std::string option_name_from_command_line = treatAsOptionName(command_line_argument);
      if (option.name() == option_name_from_command_line) {
         return command_line_arguments_[++argument_idx];
      }
   }

   return "";
}

std::string ProgramOptions::treatAsOptionName(const std::string &command_line_argument) const {
  if (command_line_argument.size() <= markOfCommandLineArgument().size()) {
    return command_line_argument;
  }

  return command_line_argument.substr(markOfCommandLineArgument().size());
}

std::string ProgramOptions::get_value_of_option(const std::string &option_name) {
   auto found_option = mandatory_options_.find(option_name);
   if (found_option == mandatory_options_.end()) {
      throw std::logic_error("option '" + option_name + "' is not defined.\n" + generate_user_guide());
   }

   if (found_option->value().empty()) {
      throw std::runtime_error("option '" + option_name + "' is not set, but should be.\n" + generate_user_guide());
   }

   return found_option->value();
}

bool ProgramOptions::detected_that_user_wants_to_see_guide() const noexcept {
   return need_to_show_user_guide_;
}

std::string ProgramOptions::generate_user_guide() const noexcept {
   std::string user_guide;
   user_guide += "usage:\n";
   user_guide += "./";
   user_guide += programName();
   user_guide += ' ';
   for (const auto& option: mandatory_options_) {
      user_guide += "<";
      user_guide += treatAsCommandLineArgument(option.name());
      user_guide += " <";
      user_guide += option.description();
      user_guide += ">> ";
   }
   user_guide += '\n';

   return user_guide;
}

std::string ProgramOptions::programName() const {
   return command_line_arguments_[0];
}

std::string ProgramOptions::treatAsCommandLineArgument(const std::string &option_name) const {
   return markOfCommandLineArgument() + option_name;
}

const std::string& ProgramOptions::markOfCommandLineArgument() const noexcept {
   static const std::string MARK_OF_COMMAND_LINE_ARGUMENT = "--";
   return MARK_OF_COMMAND_LINE_ARGUMENT;
}



