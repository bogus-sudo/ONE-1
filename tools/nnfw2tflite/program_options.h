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

#ifndef NNFW_PROGRAM_OPTIONS_H
#define NNFW_PROGRAM_OPTIONS_H

#include <string>
#include <set>

class Option {
public:
   Option(const std::string& name);

   const std::string& name() const noexcept;

   Option& description(const std::string& its_description);
   const std::string& description() const noexcept;

   void value(const std::string& its_value) const;
   const std::string& value() const noexcept;

private:
   std::string its_name_;
   std::string its_description_;
   mutable std::string its_value_;
};


class ProgramOptions {
public:
   ProgramOptions(const char* args[], size_t number_of_args);

   ProgramOptions& define(const Option& option_name);
   std::string get_value_of_option(const std::string& option_name);

   bool detected_that_user_wants_to_see_guide() const noexcept;
   std::string generate_user_guide() const noexcept;

private:
   std::string programName() const;
   std::string getFromCommandLineArgumentsValueOf(const Option& option) const;
   std::string treatAsOptionName(const std::string& command_line_argument) const;
   std::string treatAsCommandLineArgument(const std::string& command_line_argument) const;
   const std::string& markOfCommandLineArgument() const noexcept;

private:
   std::set<Option> mandatory_options_;
   const char** command_line_arguments_;
   size_t number_of_command_line_arguments_;
   bool need_to_show_user_guide_;
};

#endif //NNFW_PROGRAM_OPTIONS_H
