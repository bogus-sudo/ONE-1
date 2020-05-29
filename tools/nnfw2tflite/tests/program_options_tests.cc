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

#include "gtest/gtest.h"

#include "program_options.h"

struct ProgramOptionsTests: public testing::Test
{
};

TEST_F(ProgramOptionsTests, program_options_can_automatically_generate_help_for_program) {
   const char* fake_args[1] = {"program1"};
   int number_of_fake_args = 1;
   ProgramOptions options(fake_args, number_of_fake_args);

   options
     .define(Option("option1").description("need for bla-bla"))
     .define(Option("option2").description("need for bla-bla too"));

   ASSERT_STREQ(options.generate_user_guide().c_str(),
      "usage:\n"
      "./program1 "
      "<--option1 <need for bla-bla>> "
      "<--option2 <need for bla-bla too>> \n");
}

TEST_F(ProgramOptionsTests, program_options_can_detect_if_user_wants_to_see_guide)
{
   const char *fake_args[] = {"program1", "--help"};
   int number_of_fake_args = 2;
   ProgramOptions options(fake_args, number_of_fake_args);

   ASSERT_TRUE(options.detected_that_user_wants_to_see_guide());
}

TEST_F(ProgramOptionsTests,
        program_options_can_detect_if_user_wants_to_see_guide_and_and_define_short_key_for_that)
{
   const char *fake_args[] = {"program1", "-h"};
   int number_of_fake_args = 2;
   ProgramOptions options(fake_args, number_of_fake_args);

   ASSERT_TRUE(options.detected_that_user_wants_to_see_guide());
}

TEST_F(ProgramOptionsTests, program_options_should_signalize_if_user_dont_want_to_see_guide)
{
   const char *fake_args[] = {"program1", "--any-option", "its_value"};
   int number_of_fake_args = 3;
   ProgramOptions options(fake_args, number_of_fake_args);

   ASSERT_FALSE(options.detected_that_user_wants_to_see_guide());
}


TEST_F(ProgramOptionsTests,
        program_options_should_signalize_if_user_dont_want_to_see_guide_if_there_is_no_options_at_all)
{
   const char *fake_args[] = {"program1"};
   int number_of_fake_args = 1;
   ProgramOptions options(fake_args, number_of_fake_args);

   ASSERT_FALSE(options.detected_that_user_wants_to_see_guide());
}


TEST_F(ProgramOptionsTests, program_options_can_give_option_value_on_demand)
{
   const char* fake_args[] = {"program1", "--bla", "bla", "--option1", "value_of_option1"};
   int number_of_fake_args = 5;
   ProgramOptions options(fake_args, number_of_fake_args);

   options.define(Option("option1"));

   ASSERT_STREQ(options.get_value_of_option("option1").c_str(),"value_of_option1");
}

TEST_F(ProgramOptionsTests,
        program_options_must_generate_an_error_if_in_command_line_arguments_there_is_no_value_for_defined_option)
{
   const char* fake_args[] = {"program1", "--bla", "bla", "--option1", "value_of_option1"};
   int number_of_fake_args = 5;
   ProgramOptions options(fake_args, number_of_fake_args);

   options
   .define(Option("option1"))
   .define(Option("option2"));

   ASSERT_THROW(options.get_value_of_option("option2"), std::runtime_error);
}

TEST_F(ProgramOptionsTests,
       program_options_must_generate_an_error_there_is_no_option_with_given_name)
{
   const char* fake_args[] = {"program1", "--option1", "value_of_option1"};
   int number_of_fake_args = 3;
   ProgramOptions options(fake_args, number_of_fake_args);

   options.define(Option("option1"));

   ASSERT_THROW(options.get_value_of_option("option3"), std::logic_error);
}
